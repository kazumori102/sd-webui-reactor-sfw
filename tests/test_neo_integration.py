import unittest
import subprocess
import sys
from pathlib import Path


class ForgeNeoIntegrationTests(unittest.TestCase):
    def test_launcher_selects_sfw_without_running_legacy_installer(self):
        # Separate process: the baseline suite replaces Forge services.
        result = subprocess.run(
            [sys.executable, "-B", "-c", "from modules.launch_utils import list_extensions; "
             "names = list_extensions('config.json'); "
             "assert 'sd-webui-reactor' not in names, names; "
             "assert 'sd-webui-reactor-sfw' in names, names"],
            cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
