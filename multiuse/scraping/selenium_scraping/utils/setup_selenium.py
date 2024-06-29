"""Setup Selenium instance."""

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.chrome.options import Options
from selenium import webdriver


class SetupSelenium:
    """Setup selenium."""

    def __init__(self, command_executor: str = "", extra_options: dict = None):
        """Initialize the class."""
        if extra_options is None:
            extra_options = {}
        self.command_executor = command_executor
        self.extra_options = extra_options

    def get_base_chrome_options(self) -> Options:
        """Get the base chrome options."""
        return Options()

    def set_chrome_options(
        self, options: Options, extra_options: dict = None
    ) -> Options:
        """Set the chrome options."""
        if extra_options is None:
            extra_options = {}
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        if extra_options:
            for key, value in extra_options.items():
                options.add_argument(f"--{key}={value}")

        return options

    def get_command_executor(self, command_executor: str = "") -> str:
        """Get the command executor.

        Parameters
        ----------
        command_executor : (str, default = "http://chrome:4444/wd/hub")
            The command executor to use.
        """
        return command_executor or "http://chrome:4444/wd/hub"

    def get_chrome_driver(
        self,
        command_executor: str,
        chrome_options: Options,
    ) -> WebDriver:
        """Get the chrome driver."""
        return webdriver.Remote(
            command_executor=command_executor, options=chrome_options
        )
