"""Module for langchain utilities"""
import sys

# Silence Langchain's deprecation warnings by pretending to be in interactive
# mode (IPython or Jupyter notebook). Those warnings cannot be turned off with
# the module "warnings".
sys.ps2 = '...: '
