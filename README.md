# mlpj: Tools for machine learning projects

Installation of the [PyPi package](https://pypi.org/project/mlpj):

  `pip install -U mlpj`

Contents of this repository:
* Utilities and convenience functions for various libraries and purposes:
  * [python_utils](mlpj/python_utils.py): for the Python standard library
    * functions for basic datatypes
    * functions for filepaths and temporary files
    * functions on input and output streams
    * functions for printing to the console
    * date functions
  * [numpy_utils](mlpj/numpy_utils.py): for `numpy`
  * [pandas_utils](mlpj/pandas_utils.py): for `pandas`
    * functions to handle dataframe columns and their names
    * functions to handle undefined values
    * other dataframe convenience functions (e.g. for special cases)
    * fast groupby transform of multiple columns
    * functions to describe the contents of dataframes and series
    * many datetime convenience functions
      * e.g. add missing days to a multi-index
  * [stats_utils](mlpj/stats_utils.py): for statistical modeling
    * negative-binomial (Gamma-Poisson) distributions and overdispersion
      estimation
  * [plot_utils](mlpj/plot_utils.py): for `matplotlib`
    * histograms
    * profile plots
  * [timeseries_utils](mlpj/timeseries_utils.py): for timeseries models
  * [ml_utils](mlpj/ml_utils.py): for `sklearn` and other standard machine
    learning libraries
    * types (Protocols) for sklearn estimators and transformers
    * Find an enclosed estimator or transformer within a meta-estimator.
    * functions to print analyses of certain kinds of trained models
    * meta-estimators and meta-transformers
* [project_utils](mlpj/project_utils.py): project management utilities
  * [actions_looper](mlpj/actions_looper.py): Execute selected parts of your
    program based on persisted results of earlier steps. Together with the
    functionality mentioned below, this is meant as an alternative to Jupyter
    notebooks that integrates more seamlessly with reuse of code and
    test-driven development (TDD) while still being fairly interactive.
  * [result_display](mlpj/result_display.py): Collect textual and numerical
    results and plots on HTML pages.
