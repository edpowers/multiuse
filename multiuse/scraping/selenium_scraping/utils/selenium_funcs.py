"""Functions for use in selenium."""

import os
from datetime import datetime

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait

from multiuse.scraping.selenium_scraping.utils import (
    selenium_save_to_file,
    setup_selenium,
)


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

    def wait_for_page_load(self, driver: WebDriver, timeout: int = 10) -> bool:
        """
        Wait for the page to load.

        Returns:
        bool: True if page loaded successfully, False otherwise.
        """
        try:
            WebDriverWait(driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            return True
        except TimeoutException:
            return False

    def wait_for_page_load_with_screenshot(
        self, driver: WebDriver, timeout: int = 10
    ) -> None:
        """
        Wait for the page to load. If it fails, take a screenshot.
        """
        if not self.wait_for_page_load(driver, timeout):
            screenshot_path = self.take_screenshot(driver)
            print(f"Page load timed out. Screenshot saved at: {screenshot_path}")

    def take_screenshot(self, driver: WebDriver, directory: str = "screenshots") -> str:
        """
        Take a screenshot and save it to the specified directory.

        Returns:
        str: Path to the saved screenshot.
        """
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(directory, filename)

        # Take and save the screenshot
        driver.save_screenshot(filepath)

        return filepath

    def get_url(self, driver: WebDriver) -> str:
        return driver.current_url

    def navigate_to_url_and_save_html(
        self, webdriver_remote: WebDriver, url: str, fpath: str = ""
    ) -> None:
        """Navigate to a URL and save the HTML content to a file."""
        self.navigate_to_url(webdriver_remote, url)

        # Wait for the page to load completely
        if not self.wait_for_page_load(webdriver_remote):
            filepath = self.take_screenshot(webdriver_remote)
            print(f"Page load timed out. Screenshot saved at: {filepath}")
            raise TimeoutException("Page load timed out.")

        # Get the page source
        html_content = self.get_page_source(webdriver_remote)

        # Save the HTML content to a file
        fpath = fpath or self.get_url_path(self.get_url(webdriver_remote))

        # Save the HTML content to a file
        self.make_parent_path(fpath)

        self.write_html_to_file(html_content, fpath)

        print(f"HTML content saved to: {fpath}")
