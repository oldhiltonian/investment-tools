import datetime as dt
import yfinance as yf
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter
from scipy.stats import linregress
from pathlib import Path
import os
from typing import Dict, Tuple, List


yf.pdr_override()


class Plots:
    """
    A class to create trend plots and export them as a PDF.

    Attributes:
        ticker (str): The stock ticker of the company to analyse.
        period (str): The period of data to analyse.
        metrics (pd.DataFrame): A DataFrame containing financial metrics to plot.
        limit (int): The maximum number of metrics to plot.
        filing_dates (list): A list of dates associated with the financial filings for the company.

    Methods:
        set_limit(limit: int) -> int:
            Set the limit to the maximum number of metrics to plot.
        get_spacing() -> int:
            Get the spacing for the x-axis labels.
        filter_x_labels(x_labels, spacing) -> list:
            Filter the x-axis labels to maintain a clean plot.
        generate_x_labels() -> list:
            Generate the x-axis labels for the plot.
        calculate_subplots_shape(metrics_container) -> Tuple[int, int]:
            Calculate the subplot shape for each metric.
        select_subplot(counter: int, subplots: plt.subplots) -> plt.subplot:
            Select the appropriate subplot.
        get_y_data(metric) -> pd.Series:
            Get the y-axis data for the plot.
        get_y_units(metric_type, metric) -> str:
            Get the units for the metric to label the y-axis.
        plot_data_on_axis(axis, data) -> plt.axis:
            Plot the data on the specified axis.
        get_linear_coeffs(x, y) -> Tuple[float, float, float]:
            Calculate the slope, intercept, and R-squared value for the linear trendline.
        generate_linear_series(x, slope, intercept) -> np.ndarray:
            Generate the linear trendline series.
        plot_linear_trend(axis, x, y, r2) -> plt.axis:
            Plot the linear trendline and R-squared value.
        plot_metrics():
            Plot the financial metrics and trendlines.
        generate_save_path_object(file=False) -> Path:
            Generate the file path object for the exported PDF.
        create_path_from_object(path_object):
            Create the path specified in the path object.
        make_bin_folder():
            Create a bin folder to store temporary files.
        make_pdf_title_page():
            Create the title page of the PDF.
        make_pdf_charts():
            Create the trend plot pages of the PDF.
        combine_and_export_pdfs(export_path):
            Combine the title page and trend plot pages and export the PDF.
        _export_charts_pdf():
            Export the trend plots as a PDF.

    """

    def __init__(
        self,
        ticker: str,
        period: str,
        metrics: pd.DataFrame,
        limit: str,
        filing_dates: pd.Index,
    ):
        """
        Creates an instance of the `Plots` class.

        Args:
            ticker (str): The ticker symbol of the company.
            period (str): The reporting period of the financial data.
            metrics (pd.DataFrame): The financial metrics of the company.
            limit (int): The maximum number of metrics to display.
            filing_dates (List[str]): The filing dates of the financial data.

        Attributes:
            ticker (str): The ticker symbol of the company.
            period (str): The reporting period of the financial data.
            metrics (pd.DataFrame): The financial metrics of the company.
            filing_dates (List[str]): The filing dates of the financial data.
            limit (int): The maximum number of metrics to display.
            metric_units_dict (dict): A dictionary containing the units for each metric.
            plots (List[matplotlib.figure.Figure]): A list containing the plotted charts.
        """
        self.ticker = ticker
        self.period = period
        self.metrics = metrics
        self.filing_dates = filing_dates
        self.limit = self.set_limit(limit)
        self.metric_units_dict = self.get_metric_units_dict()
        self.plots = []
        self.plot_metrics()

    @staticmethod
    def get_metric_units_dict() -> Dict:
        """
        Returns the dictionary that maps the financial metric with its corresponding unit.

        Args:
            None

        Returns:
            Dict: the mapping dictionary.
        """
        metric_units_dict = {
            "Stock Evaluation Ratios": {
                "eps": "$/share",
                "eps_diluted": "$/share",
                "PE_high": "x",
                "PE_low": "x",
                "bookValuePerShare": "$/share",
                "dividendPayoutRatio": "x",
                "cashPerShare": "$/share",
                "ebitdaratio": "x",
            },
            "Profitability Ratios": {
                "grossProfitMargin": "Gross Profit/Sales",
                "operatingProfitMargin": "Operating Profit/Sales",
                "pretaxProfitMargin": "Pretax Profit/Sales",
                "netProfitMargin": "Net Profit/Sales",
                "ROIC": "x",
                "returnOnEquity": "x",
                "returnOnAssets": "x",
            },
            "Debt & Interest Ratios": {
                "interestCoverage": "EBIT/Interest Expense",
                "fixedChargeCoverage": "EBIT/Fixed Charges",
                "debtToTotalCap": "LT_Debt/Total Capitalization",
                "totalDebtRatio": "Total Debt/Total Assets",
            },
            "Liquidity Ratios": {
                "currentRatio": "Current Assets/Current Liabilities",
                "quickRatio": "Quick Assets/Current Liabilities",
                "cashRatio": "Cash/Current Liabilities",
            },
            "Efficiency Ratios": {
                "totalAssetTurnover": "Sales/Total Assets",
                "inventoryToSalesRatio": "Inventory/Sales",
                "inventoryTurnoverRatio": "Sales/Inventory",
                "inventoryTurnoverInDays": "Days",
                "accountsReceivableToSalesRatio": "AccountsReceivable/Sales",
                "receivablesTurnover": "Sales/AccountsReceivable",
                "receivablesTurnoverInDays": "Days",
            },
        }

        return metric_units_dict

    def set_limit(self, limit: int) -> int:
        """
        Sets the maximum number of metrics to display in the plots.

        Args:
            limit (int): The maximum number of metrics to display.

        Returns:
            int: The maximum number of metrics to display.
        """
        if limit > len(self.metrics):
            return len(self.metrics)
        else:
            return limit

    def get_spacing(self) -> int:
        """
        Determines the spacing between x-tick labels for the plots, based on the number of metrics plotted.

        Returns:
            int: The spacing between x-tick labels.
        """
        if self.limit < 10:
            return 2
        elif self.limit < 20:
            return 4
        else:
            return 6

    def filter_x_labels(self, x_labels, spacing) -> List[str]:
        """
        Filters the x-tick labels for the plots, based on the specified spacing.

        Args:
            x_labels (list): The list of x-tick labels to be filtered.
            spacing (int): The spacing between x-tick labels.

        Returns:
            list: The filtered list of x-tick labels.
        """
        return [x_labels[i] if i % spacing == 0 else " " for i in range(len(x_labels))]

    def generate_x_labels(self) -> List[str]:
        """
        Generates the x-tick labels for the plots, based on the metrics data and the specified period.

        Returns:
            list: The list of x-tick labels for the plots.
        """
        return [
            "-".join(str(i).split("-")[1:]) for i in self.metrics.index[-self.limit :]
        ]

    def calculate_subplots_shape(self, metrics_container: List) -> Tuple[int, int]:
        """
        Calculates the number of rows and columns required for the subplot grid, based on the number of
            metrics to be plotted.

        Args:
            metrics_container (list): The list of metrics to be plotted.

        Returns:
            tuple: A tuple containing the number of rows and columns required for the subplot grid.
        """
        nplots = len(metrics_container)
        nrows = -(-nplots // 2)
        ncols = 2
        return nrows, ncols

    def select_subplot(self, counter: int, subplots: plt.subplots) -> plt.subplot:
        """
        Selects a subplot from the grid of subplots, based on the current counter and the subplots grid.

        Args:
            counter (int): The current counter value.
            subplots (matplotlib.figure.Subplot): The grid of subplots.

        Returns:
            plt.subplot: The selected subplot.
        """
        i, j = counter // 2, counter % 2
        return subplots[i][j]

    def get_y_data(self, metric: str) -> pd.Series:
        """
        Extracts the y-axis data for the specified metric.

        Args:
            metric (str): The name of the metric.

        Returns:
            pd.Series: The y-axis data for the specified metric.
        """
        return self.metrics[metric][-self.limit :]

    def get_y_units(self, metric_type: str, metric: str) -> str:
        """
        Extracts the unit of measure for the y-axis for the specified metric.

        Args:
            metric_type (str): The category of the metric.
            metric (str): The name of the metric.

        Returns:
            str: The unit of measure for the y-axis for the specified metric.
        """
        return self.metric_units_dict[metric_type][metric]

    def plot_data_on_axis(self, axis: plt.axis, data: pd.Series) -> plt.axis:
        """
        Plots the data trend for the specified metric on the specified axis.

        Args:
            axis (plt.subplot): The axis on which to plot the data.
            data (dict): A dictionary containing the plotting data, including the metric data, labels,
                and units of measure.
        """
        axis.plot(data["y"], label="data")
        axis.set_title(data["metric"])
        axis.set_xticks(data["x_true"])
        axis.set_xticklabels(data["x_labels"])
        axis.set_ylabel(data["y_label"])
        axis = self.scale_y_axis(axis, data["metric"])
        return axis

    def scale_y_axis(self, axis: plt.axis, data_str: str) -> plt.axis:
        """
        Scales the y-axis of a given axis object based on the value of the
        data_str parameter.

        Args:
            axis (plt.axis): The axis to modify.
            data_str (str): A string indicating which type of data is plotted on the y-axis.

        Returns:
            plt.axis: The modified axis object, with the y-axis limits set according to the data_str parameter.
        """
        if data_str == "PE_high":
            y_bounds = [0, 40]
        elif data_str == "PE_low":
            y_bounds = [0, 40]
        elif data_str in [
            "dividendPayoutRatio",
            "ebitdaratio",
            "debtToTotalCap",
            "totalDebtRatio",
        ]:
            y_bounds = [0, 1]
        else:
            return axis
        axis.set_ylim(y_bounds)
        return axis

    def get_linear_coeffs(
        self, x: pd.Series, y: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Calculate the linear regression coefficients and coefficient of determination (R-squared)
        for the given x and y data points.

        Args:
            x (List[float]): List of x data points.
            y (List[float]): List of y data points.

        Returns:
            Tuple[float, float, float]: A tuple containing the slope, intercept, and R-squared value.

        Calculates the linear regression coefficients for the given x and y data points using the
        `linregress` function from the `scipy.stats` module. The slope, intercept, and R-squared
        value are returned as a tuple.
        """
        slope, intercept, r_value, _, _ = linregress(x, y)
        return slope, intercept, r_value ** 2

    def generate_linear_series(
        self, x: pd.Series, slope: float, intercept: float
    ) -> pd.Series:
        """
        Returns a pandas Series containing the y-values of a linear series with the given x-values,
        slope, and intercept.

        Args:
            x (pandas.Series): The x-values of the linear series.
            slope (float): The slope of the linear series.
            intercept (float): The intercept of the linear series.

        Returns:
            pandas.Series: The y-values of the linear series.
        """
        return slope * x + intercept

    def plot_linear_trend(
        self, axis: plt.axis, x: pd.Series, y: pd.Series, r2: float
    ) -> plt.axis:
        """
        Plots linear regression line on given axis with r-squared value in legend.

        Args:
            axis (matplotlib axis): axis object on which the line will be plotted.
            x (np.ndarray): x-values for the data.
            y (np.ndarray): y-values for the data.
            r2 (float): r-squared value for the linear regression.

        Returns:
            matplotlib axis: axis object with the plotted line.
        """
        axis.plot(x, y, alpha=0.5, linestyle="--", label="linear trend")
        axis.plot([], [], " ", label=f"R2: {r2:.2f}")  # Adding R2 value to legend
        return axis

    def plot_metrics(self) -> None:
        """
        The plot_metrics method generates plots of financial metrics for a given company. The method first
            generates the necessary subplots to hold each metric type. Each metric type is then plotted on
            an individual axis within the metric type's subplot. Each metric's trend data is gathered, and
            its trend is plotted on the selected axis. A linear trendline is also plotted along with its
            corresponding R2 value. Finally, each metric type's subplot is given a title and added to the
            list of plots.

        Args:
        self (Plots): The Plots instance.

        Returns:
        None. The method appends the generated plots to the Plots instance.
        """
        spacing = self.get_spacing()
        x_labels = self.filter_x_labels(self.generate_x_labels(), spacing)

        for metric_type in self.metric_units_dict.keys():
            # creating subplot axes to hold all metrics of each metric type
            metrics = self.metric_units_dict[metric_type].keys()
            nrows, ncols = self.calculate_subplots_shape(metrics)
            fig, ax = plt.subplots(nrows, ncols, figsize=(11.7, 8.3))

            # plotting each metric trend on an axis
            for counter, metric in enumerate(metrics):
                # axis selection
                axis = self.select_subplot(counter, ax)
                # gathering data trend for each metric
                y = self.get_y_data(metric)
                y_label = self.get_y_units(metric_type, metric)
                x_true = y.index
                x_dummy = range(len(y))

                # plotting the data
                plotting_data = {
                    "metric": metric,
                    "y": y,
                    "y_label": y_label,
                    "x_true": x_true,
                    "x_dummy": x_dummy,
                    "x_labels": x_labels,
                }
                self.plot_data_on_axis(axis, plotting_data)

                # plotting the linear trendline and R2
                slope, intercept, r2_value = self.get_linear_coeffs(x_dummy, y)
                y_linear = self.generate_linear_series(x_dummy, slope, intercept)
                self.plot_linear_trend(axis, x_dummy, y_linear, r2_value)
                axis.legend(loc="upper right", frameon=False, fontsize=8)

            # formatting and appending the figure
            fig.suptitle(metric_type)
            fig.tight_layout()
            self.plots.append(fig)

    def generate_save_path_object(self, file: bool = False) -> Path:
        """
        Generates the path to the PDF file where the charts will be saved.

        Args:
            file (bool, optional): Determines whether or not the filename should be appended to the file path.
                Defaults to False.

        Returns:
            PosixPath: The file path object.
        """
        date = str(dt.datetime.now()).split()[0]
        end_date = self.filing_dates[-1]
        start_date = self.filing_dates[-self.limit]
        file_name = (
            f"{self.ticker}_{self.period}_{str(start_date)}_to_{str(end_date)}.pdf"
        )
        file_path = (
            Path.cwd() / "data" / "Company_Analysis" / date / self.ticker / self.period
        )
        return file_path / file_name if file else file_path

    def create_path_from_object(self, path_object: Path) -> None:
        """
        Creates a directory at the given path if it does not already exist.

        Args:
            path_object (Path): The path to create the directory.
        """
        try:
            os.makedirs(path_object)
        except FileExistsError:
            pass

    def make_bin_folder(self) -> None:
        """
        Creates a 'bin' folder if it does not already exist.
        """
        try:
            os.mkdir("bin")
        except FileExistsError:
            pass

    def make_pdf_title_page(self) -> None:
        """
        Generates a PDF title page using the ticker and period of the company, and saves it in the 'bin' folder.
        """
        self.title_path = "bin/title.pdf"
        self.charts_path = "bin/charts.pdf"
        title_message = f"Financial Ratio Trends for {self.ticker}"
        title_page = canvas.Canvas(self.title_path)
        title_page.drawString(210, 520, title_message)
        title_page.save()

    def make_pdf_charts(self) -> None:
        """
        Generates individual PDFs for each figure in self.plots, and saves them in the 'bin' folder.
        """
        self.charts_path = "bin/charts.pdf"
        with PdfPages(self.charts_path) as pdf:
            for figure in self.plots:
                pdf.savefig(figure)

    def combine_and_export_pdfs(self, export_path: Path) -> None:
        """
        Combines the PDFs generated in make_pdf_title_page and make_pdf_charts, and exports them to the given path.

        Args:
            export_path (str): The file path to export the combined PDF to.
        """
        with open(self.title_path, "rb") as f1:
            with open(self.charts_path, "rb") as f2:
                pdf1 = PdfReader(f1, "rb")
                pdf2 = PdfReader(f2, "rb")
                pdf_output = PdfWriter()
                for page_num in range(len(pdf1.pages)):
                    pdf_output.add_page(pdf1.pages[page_num])
                for page_num in range(len(pdf2.pages)):
                    pdf_output.add_page(pdf2.pages[page_num])
                with open(export_path, "wb") as output_file:
                    pdf_output.write(output_file)

    def _export_charts_pdf(self) -> None:
        """
        Generates and exports a combined PDF file of the financial ratio trend plots to a file path
        specified by generate_save_path_object.
        """
        path_object = self.generate_save_path_object(file=False)
        self.create_path_from_object(path_object)
        self.make_bin_folder()
        self.make_pdf_title_page()
        self.make_pdf_charts()
        path_object = self.generate_save_path_object(file=True)
        self.combine_and_export_pdfs(path_object)
