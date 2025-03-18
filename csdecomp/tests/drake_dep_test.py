import unittest
import numpy as np
print(f"numpy version {np.__version__}")

import sys

class TestBasic(unittest.TestCase):
    def test_drake_import(self):
        from pydrake.all import (RobotDiagramBuilder,
                         LoadModelDirectives,
                         ProcessModelDirectives,
                         SceneGraphCollisionChecker,
                         QueryObject, 
                         SceneGraphInspector)
  

if __name__ == "__main__":
  unittest.main()