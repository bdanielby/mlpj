# mlpj: Tools for machine learning projects

Contents of this repository:
* Utilities and convenience functions for various Python libraries:
  * [python_utils](mlpj/python_utils.py): for the Python standard library
  * [numpy_utils](mlpj/numpy_utils.py): for `numpy`
  * [pandas_utils](mlpj/pandas_utils.py): for `pandas`
  * [plot_utils](mlpj/plot_utils.py): for `matplotlib`
  * [timeseries_utils](mlpj/timeseries_utils.py): for timeseries models
  * `torch_utils`: for `torch` (coming soon)
* [project_utils](mlpj/project_utils.py): project management utilities
  * [actions_looper](mlpj/actions_looper.py): Execute selected parts of your
    program based on persisted results of earlier steps.
  * [result_display](mlpj/result_display.py): Collect textual and numerical
    results and plots on HTML pages.
