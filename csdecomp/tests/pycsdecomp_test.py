import unittest
import pycsdecomp

import numpy as np
import sys

class TestBasic(unittest.TestCase):

  def test_link(self):
    link = pycsdecomp.Link()
    link.name = "elbow"
    self.assertEqual(link.name, "elbow")
    print('hello world my python bindings are working :))')
    print(f" numpy version {np.__version__}")
    print(f"location of python interpreter {sys.executable}")
    
if __name__ == "__main__":
  unittest.main()