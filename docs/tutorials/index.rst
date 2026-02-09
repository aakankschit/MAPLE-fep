Interactive Tutorials
=====================

These interactive tutorials demonstrate MAPLE's capabilities using real benchmark datasets.
The tutorials are built with `Marimo <https://marimo.io/>`_, a modern reactive Python notebook.

.. note::

   These tutorials are fully interactive. You can modify the code and see results update in real-time.
   To run the tutorials locally, install Marimo with ``pip install marimo`` and run:

   .. code-block:: bash

      marimo edit docs/tutorials/getting_started.py


Getting Started Tutorial
------------------------

This tutorial walks you through the basics of MAPLE:

- Loading benchmark FEP datasets
- Understanding the data structure (nodes and edges)
- Training a NodeModel with Bayesian inference
- Analyzing model results and performance metrics
- Visualizing predictions vs experimental values

.. raw:: html

   <div style="margin: 2em 0;">
   <a href="getting_started.html" class="btn btn-primary" style="padding: 10px 20px; background-color: #2980b9; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">
   View Interactive Tutorial
   </a>
   </div>

The tutorial uses the **CDK8 (Cyclin-dependent kinase 8)** benchmark dataset to demonstrate:

1. **Data Loading**: How to use ``FEPDataset`` to load benchmark data
2. **Model Configuration**: Setting up ``NodeModelConfig`` with appropriate priors
3. **Training**: Running MAP inference with the ``NodeModel``
4. **Analysis**: Calculating RMSE, MAE, and correlation metrics
5. **Visualization**: Creating correlation plots

Source Files
------------

The Marimo notebook source files are available in the repository:

- `getting_started.py <https://github.com/aakankschit/maple-fep/blob/main/docs/tutorials/getting_started.py>`_ - Getting Started Tutorial


Running Tutorials Locally
-------------------------

To run tutorials interactively:

1. Install MAPLE with notebook dependencies:

   .. code-block:: bash

      pip install maple-fep[notebook]

2. Navigate to the tutorials directory:

   .. code-block:: bash

      cd docs/tutorials

3. Start Marimo:

   .. code-block:: bash

      marimo edit getting_started.py

This will open an interactive notebook in your browser where you can modify code and see results update reactively.
