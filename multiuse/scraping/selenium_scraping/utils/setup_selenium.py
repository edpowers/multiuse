"""Setup Selenium instance."""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webdriver import WebDriver


class SetupSelenium:
    """Setup selenium.

    Example
    -------
    selenium_instance = SetupSelenium()
    driver = selenium_instance.setup_webdriver()
    """

    def __init__(self, command_executor: str = "", extra_options: dict = {}):
        self.command_executor = command_executor
        self.extra_options = extra_options

    @classmethod
    def setup_webdriver(cls, command_executor: str = "") -> WebDriver:
        """Setup the webdriver."""
        instance = cls()

        instance.command_executor = instance._get_command_executor(command_executor)
        instance.extra_options = {}

        chrome_options = instance._get_base_chrome_options()
        chrome_options = instance._set_chrome_options(
            chrome_options,
            extra_options=instance.extra_options,
        )

        return instance._get_chrome_driver(
            command_executor=instance.command_executor, chrome_options=chrome_options
        )

    def _get_base_chrome_options(self) -> Options:
        """Get the base chrome options."""
        return Options()

    def _set_chrome_options(
        self, options: Options, extra_options: dict = {}
    ) -> Options:
        """Set the chrome options."""
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        if extra_options:
            for key, value in extra_options.items():
                options.add_argument(f"--{key}={value}")

        return options

    def _get_command_executor(self, command_executor: str = "") -> str:
        """Get the command executor.

        Parameters
        ----------
        command_executor : (str, default = "http://chrome:4444/wd/hub")
            The command executor to use.
        """
        return command_executor or "http://chrome:4444/wd/hub"

    def _get_chrome_driver(
        self,
        command_executor: str,
        chrome_options: Options,
    ) -> WebDriver:
        """Get the chrome driver."""
        return webdriver.Remote(
            command_executor=command_executor, options=chrome_options
        )
