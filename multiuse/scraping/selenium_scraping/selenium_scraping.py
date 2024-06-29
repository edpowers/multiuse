"""Class for setup + scraping of selenium."""

from pydantic import BaseModel
from selenium.webdriver.remote.webdriver import WebDriver

from multiuse.scraping.selenium_scraping.utils.selenium_funcs import SeleniumFuncs


class SeleniumKwargs(BaseModel):
    """Keyword arguments for SeleniumScraping."""

    fpath: str = ""


class SeleniumScraping(SeleniumFuncs):
    def __init__(self, **kwargs: SeleniumKwargs) -> None:
        super().__init__(**kwargs)

    def wrap_navigate_to_url_and_save_html(
        self, webdriver_remote: WebDriver, url: str, fpath: str = ""
    ) -> None:
        """Wrap the navigate_to_url_and_save_html function."""
        try:
            self.navigate_to_url_and_save_html(webdriver_remote, url, fpath)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Close the browser
            webdriver_remote.quit()
