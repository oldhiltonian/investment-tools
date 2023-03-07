investment-tools
================
A set of tools to help pull, analyse and visualise company financial data from the Financial Modelling Prep (FMP) API. The tool includes functionality to assess company health and future prospects, and can be used as an screener to quickly determine if a company warrants a closer, manual analysis.

Needless to say, you should not make investments based on this tool's recommendations. It is merely here to help you filter through the bad investment opportunities such that you may focus your attention on the good ones.

Prerequisites
-------------
Please head over to the `FMP website <https://site.financialmodelingprep.com/developer/docs/dashboard>`_ and register in order to obtain an API key that you can use to make API requests.

The requirements for using this library are as follows:

- pandas == 1.4.4
- numpy == 1.21.5
- scipy == 1.9.1
- yfinance == 0.2.10
- pandas_datareader == 0.10.0
- matplotlib == 3.5.2
- reportlab == 3.6.12
- pyarrow == 11.0.0
- PyPDF2 == 3.0.1

Installing
----------
This code is a simple set of classes that can be run on either Windows or Mac. No installation is required other than getting the files into a local directory and then executing scripts from the project root directory.

Getting Started
---------------
Please see the "Example Use" Jupyter notebook in the root directory for more robust demonstrations on how to use the tools.

The main class that is meant to be interacted with is the Company class. An analysis can be constructed by simply calling the Company constructor and passing the appropriate arguments for the company ticker symbol and your API key.\n

:code: `>>> Company('AAPL', YOUR_API_KEY)`
Optional arguments are:

data: 'online' fetches from the online API, 'local' fetched previously obtained API data from disk.
period: 'annual' or 'quarter' determines the type of financial statement data that is loaded.
limit: detemines the number of historical periods that data is obtained for.
verbose: setting to True will force the output of all visualisations regardless of company health.
Contributing
Please feel free to contact me on this GitHub if you would like to contribute. This is a personal project that I use to help my own investment workflow, so there is no guarentee that I will accept the proposed changes. I am however very open to hearing your feedback and your experience with the tool.

Authors
-------
**JFD ** -  `oldhiltonian<https://github.com/oldhiltonian>`_

License
-------
This project is unlicensed and is completely open source.