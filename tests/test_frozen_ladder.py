import ast
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from frozen_ladder import DEFAULT_ROOT,id_hash,load_ladder
from materialize_frozen_tier import chunk_body,materialize


class FrozenLadderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bedrock,cls.ocean,cls.checks=load_ladder()

    def test_historical_sets_and_target(self):
        self.assertEqual(len(self.checks),28)
        order=self.bedrock+self.ocean[:41443]
        self.assertEqual(len(set(order)),42587)
        self.assertEqual(id_hash(sorted(order)),'cc70d16d9754ff127b2c6b3f8d5a2b08dab725e400e54ff9b51ecca7e2138e2e')

    def test_order_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'manifests';shutil.copytree(DEFAULT_ROOT.parent,root)
            ids=list(self.ocean);ids[0],ids[1]=ids[1],ids[0]
            with gzip.open(root/'frozen/ocean_order.txt.gz','wt',encoding='utf-8') as f:f.write('\n'.join(ids)+'\n')
            with self.assertRaisesRegex(ValueError,'order mismatch'):load_ladder(root/'frozen')

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'manifests';shutil.copytree(DEFAULT_ROOT.parent,root)
            ids=list(self.bedrock);ids[1]=ids[0]
            (root/'frozen/bedrock_dsids.txt').write_text('\n'.join(ids)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Repeated identifier'):load_ladder(root/'frozen')

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'keep.txt';p.write_text('original',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'already exists'):materialize('unused','unused',temp)
            self.assertEqual(p.read_text(encoding='utf-8'),'original')

    def test_bad_historical_reference_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'manifests';shutil.copytree(DEFAULT_ROOT.parent,root)
            p=root/'tier_manifest_summary.json';refs=json.loads(p.read_text(encoding='utf-8'))
            refs[0]['manifest_sha256']='0'*64;p.write_text(json.dumps(refs),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Historical document set mismatch'):load_ladder(root/'frozen')

    def test_chunking_matches_unchanged_original_function(self):
        class Encoding:
            def encode(self,text,**kwargs):return list(text)
            def decode(self,tokens):return ''.join(tokens)
        module=ast.parse((ROOT/'scripts/pack_corpus_enterprise.py').read_text(encoding='utf-8'))
        node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='chunk_body')
        ns={'enc':Encoding(),'CHUNK_SIZE':1200,'CHUNK_OVERLAP':100}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'historical_chunk_body','exec'),ns)
        for size in [0,1,1199,1200,1201,2300,2301,3401]:
            text=''.join(chr(65+i%26) for i in range(size))
            with self.subTest(size=size):self.assertEqual(chunk_body(text,Encoding()),ns['chunk_body'](text))

    def test_cli_export_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'ids'
            command=[sys.executable,str(ROOT/'scripts/frozen_ladder.py'),'--tier','42587','--out',str(target)]
            subprocess.run(command,check=True,capture_output=True)
            ids=(target/'ordered_dsids.txt').read_text(encoding='utf-8').splitlines()
            self.assertEqual(ids,self.bedrock+self.ocean[:41443])
            before=(target/'manifest.json').read_bytes()
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual((target/'manifest.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
