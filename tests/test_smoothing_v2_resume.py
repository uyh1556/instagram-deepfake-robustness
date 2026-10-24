"""An interrupted v2 diagonal run resumes under its own saved config when only this trainer's digest differs."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import beauty_smoothing_v2 as v2
import ffpp_c23_smoothing_v2_diagonal as diagonal
import test_beauty_smoothing_v2 as v2_fixture
import test_smoothing_v2_diagonal as diag_fixture


class ResumeTests(unittest.TestCase):
    setUp = diag_fixture.DiagonalTests.setUp
    tearDown = diag_fixture.DiagonalTests.tearDown
    train_args = diag_fixture.DiagonalTests.train_args
    build_caches = diag_fixture.DiagonalTests.build_caches

    def test_resume_after_trainer_edit(self):
        with patch.object(v2, 'SkinParser', v2_fixture.FakeParser), patch.object(v2, 'full', side_effect=v2_fixture.fake_full):
            self.build_caches()
            t = self.train_args(strategies=['aug_pair'], epochs=2)
            diagonal.train(t)
            run = t.runs_root/diagonal.run_name('dinov2_b14', 'bilateral', 'aug_pair', 42)
            # Make it look interrupted by an OLDER trainer: other digest of this file, no completed.json.
            saved = json.loads((run/'config.json').read_text())
            saved['code_sha256'][Path(diagonal.__file__).name] = 'older-trainer'
            (run/'config.json').write_text(json.dumps(saved))
            state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
            state.update(identity=diagonal.evaluation.key(saved), config=saved)
            torch.save(state, run/'last.pt')
            (run/'completed.json').unlink()
            with patch.object(diagonal, 'PairedV2', side_effect=AssertionError('retrained from scratch')):
                diagonal.train(self.train_args(strategies=['aug_pair'], epochs=2))
            self.assertEqual(json.loads((run/'config.json').read_text()), saved)   # kept its own config
            done = json.loads((run/'completed.json').read_text())
            self.assertEqual(done['identity'], diagonal.evaluation.key(saved))
            self.assertEqual(len(json.loads((run/'history.json').read_text())), 2)    # resumed from last.pt, not retrained
            # A TRAINING difference still refuses.
            saved['lr_changed'] = True
            (run/'config.json').write_text(json.dumps(saved)); (run/'completed.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Existing protocol differs'):
                diagonal.train(self.train_args(strategies=['aug_pair'], epochs=2))


if __name__ == '__main__':
    unittest.main()
