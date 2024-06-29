"""Functions for use in selenium."""

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webdriver import WebDriver


from multiuse.scraping.selenium_scraping.utils import selenium_save_to_file
from multiuse.scraping.selenium_scraping.utils import setup_selenium


class SeleniumFuncs(
    selenium_save_to_file.SeleniumSaveToFile,
    setup_selenium.SetupSelenium,
):
    """Functions for use in selenium."""

    def navigate_to_url(self, driver: WebDriver, url: str) -> None:
        """Navigate to a URL."""
        driver.get(url)

    def get_page_source(self, driver: WebDriver) -> str:
        """Get the page source."""
        return driver.page_source

    def wait_for_page_load(self, driver: WebDriver) -> None:
        """Wait for the page to load."""
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

    def get_url(self, driver: WebDriver) -> str:
        return driver.current_url

    def navigate_to_url_and_save_html(
        self, webdriver_remote: WebDriver, url: str, fpath: str = ""
    ) -> None:
        """Navigate to a URL and save the HTML content to a file."""
        self.navigate_to_url(webdriver_remote, url)

        # Wait for the page to load completely
        self.wait_for_page_load(webdriver_remote)

        # Get the page source
        html_content = self.get_page_source(webdriver_remote)

        # Save the HTML content to a file
        fpath = fpath or self.get_url_path(self.get_url(webdriver_remote))

        # Save the HTML content to a file
        self.make_parent_path(fpath)

        self.write_html_to_file(html_content, fpath)
