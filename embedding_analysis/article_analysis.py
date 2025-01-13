from datetime import datetime
from dateutil.parser import isoparse
from embedding_analysis.embbedings_analysis import EmbeddingsAnalysis

import util

PINK = "\033[38;5;205m"
RESET = "\033[0m"
ORANGE = "\033[38;5;208m"
WHITE = "\033[97m"
BLUE = "\033[34m"


class ArticleAnalysis(EmbeddingsAnalysis):


    def __init__(self, newsagencies :list = [], update: bool = True, reset: bool = False,
                 debug: bool = False, batch_size: int = 100):
        if len(newsagencies) == 0:
            newsagencies = util.load_config()["data_topics"].keys()
        super().__init__(newsagencies, update, reset, debug, batch_size)

    def get_authors(self):
        return [metadata.get("author", None) for metadata in self.metadata]

    def get_titles(self):
        return [metadata.get("title", None) for metadata in self.metadata]

    def get_descriptions(self):
        return [metadata.get("description", None) for metadata in self.metadata]

    def get_urls(self):
        return [metadata.get("url", None) for metadata in self.metadata]

    def get_published_dates(self):
        return [isoparse(metadata.get("publishedAt", None)).date() if metadata.get("publishedAt", None) else None for metadata in self.metadata]

    def get_published_datetimes(self):
        return [metadata.get("publishedAt", None) for metadata in self.metadata]

    def get_contents(self):
        return [metadata.get("content", None) for metadata in self.metadata]

    def get_txt_files(self):
        return [metadata.get("txt_file", None) for metadata in self.metadata]

    def get_source_ids(self):
        return [metadata.get("source_id", None) for metadata in self.metadata]

    def get_source_names(self):
        return [metadata.get("source_name", None) for metadata in self.metadata]

    def get_source_descriptions(self):
        return [metadata.get("source_description", None) for metadata in self.metadata]

    def get_source_urls(self):
        return [metadata.get("source_url", None) for metadata in self.metadata]

    def get_source_categories(self):
        return [metadata.get("source_category", None) for metadata in self.metadata]

    def get_source_languages(self):
        return [metadata.get("source_language", None) for metadata in self.metadata]

    def get_source_countries(self):
        return [metadata.get("source_country", None) for metadata in self.metadata]
