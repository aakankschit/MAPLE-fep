Contributing to MAPLE
====================

We welcome contributions to MAPLE! This guide provides information on how to contribute to the project.

Development Setup
-----------------

Fork and Clone
~~~~~~~~~~~~~~

1. Fork the MAPLE repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/your-username/MAPLE.git
   cd MAPLE

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

1. Create a virtual environment:

.. code-block:: bash

   python -m venv maple-dev
   source maple-dev/bin/activate  # On Windows: maple-dev\Scripts\activate

2. Install in development mode:

.. code-block:: bash

   pip install -e .[dev]

3. Install pre-commit hooks:

.. code-block:: bash

   pre-commit install

Running Tests
~~~~~~~~~~~~~

Run the test suite:

.. code-block:: bash

   pytest tests/

Run tests with coverage:

.. code-block:: bash

   pytest tests/ --cov=maple --cov-report=html

Code Style
----------

We use several tools to maintain code quality:

* **Black**: Code formatting
* **flake8**: Linting
* **mypy**: Type checking

Format code:

.. code-block:: bash

   black src/ tests/

Check linting:

.. code-block:: bash

   flake8 src/ tests/

Type checking:

.. code-block:: bash

   mypy src/maple/

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

To build the documentation locally:

.. code-block:: bash

   cd docs/
   make html

The built documentation will be in ``docs/_build/html/``.

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

* Use NumPy-style docstrings for functions and classes
* Include examples in docstrings when helpful
* Update relevant documentation when adding features

Submitting Changes
------------------

1. Create a feature branch:

.. code-block:: bash

   git checkout -b feature-name

2. Make your changes and add tests
3. Ensure all tests pass and code follows style guidelines
4. Commit your changes with a clear message
5. Push to your fork and create a pull request

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

* Include a clear description of the changes
* Reference any related issues
* Ensure all CI checks pass
* Add tests for new functionality
* Update documentation as needed

Reporting Issues
----------------

When reporting bugs or requesting features:

* Use the GitHub issue tracker
* Provide a clear description and reproduction steps
* Include relevant system information
* Attach minimal examples when possible
