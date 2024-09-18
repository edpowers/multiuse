from multiuse.imports.lazy_loader_internal import LazyLoaderInternal

selenium_scraping = LazyLoaderInternal(
    "multiuse.scraping.selenium_scraping.selenium_scraping"
)

__all__ = ["selenium_scraping"]
