# Imports
import argparse
import re
import json
from pathlib import Path
import torch
from transformers import pipeline
from bs4 import BeautifulSoup
import cloudscraper
from urllib.parse import urlparse
import unicodedata
from tqdm import tqdm

class DatasetGenerator:
    """
    Generate a synthetic recipe article
        dataset with a HuggingFace model.
    Args:
        hf_model_name: Identifying name of huggingface model.
        hf_access_token: Optional HuggingFace access token.
        article_urls_file: JSON file containing urls to recipe articles.
        dataset_file: JSON file to save generated dataset to.
    """
    def __init__(self, hf_model_name: str, hf_access_token: str|None,
                 article_urls_file: str, dataset_file: str) -> None:
        article_urls_file = Path(article_urls_file)
        self._dataset_file = Path(dataset_file)
        assert all(p.suffix == '.json' for p in
                   (article_urls_file, self._dataset_file))
        self._pipe = pipeline(
            "text-generation", model=hf_model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto", token=hf_access_token)
        self._terminators = [
            self._pipe.tokenizer.eos_token_id,
            self._pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self._scraper = cloudscraper.create_scraper()
        with open(article_urls_file) as fp:
            self._urls = json.load(fp)
        self._dataset = {}

    def generate(self) -> None:
        for provider, article_urls in tqdm(self._urls.items(), unit='provider',
                                           desc='Generating Data'):
            self._dataset[provider] = {}
            for url in tqdm(article_urls, unit='article', leave=False):
                soup = BeautifulSoup(self._scraper.get(url).text, 'html.parser')
                article = soup.get_text()
                article = self._clean_article_text(article)
                ingredients, instructions = self._extract_info(article)
                sample = {'article': article, 'ingredients':
                          ingredients, 'instructions': instructions}
                parsed_url = urlparse(url)
                name = parsed_url.path[1:-1]
                self._dataset[provider][name] = sample
                break # TEST
        with open(self._dataset_file, 'w') as fp:
            json.dump(self._dataset, fp, indent=2)

    def _extract_info(self, article: str) -> tuple[str]:
        output = []
        for task in ('ingredients', 'instructions'):
            messages = [
                {"role": "system", "content": ("You are a kitchen assistant chatbot for a chef. "
                                               f"The chef wants to make the recipe from this article: {article}")},
                {"role": "user", "content": f"Provide only a bulleted list of the recipe's {task}. Do not add any other text."}]
            task_output = self._pipe(messages, max_new_tokens=1024, eos_token_id=self._terminators,
                                     do_sample=True, temperature=0.6, top_p=0.9)
            output += [task_output[0]['generated_text'][-1]['content']]
        return tuple(output)
    
    def _clean_article_text(self, article: str) -> str:
        article = re.sub(' +', ' ', article)
        article = re.sub('\n+', '\n', article)
        article = re.sub('\t+', '\t', article)
        article = unicodedata.normalize("NFKD", article)
        article = article.strip()
        return article


parser = argparse.ArgumentParser('Generate a synthetic recipe article dataset with a HuggingFace model.')
parser.add_argument('-model', default='meta-llama/Meta-Llama-3-8B-Instruct')
parser.add_argument('-token', default='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')
parser.add_argument('-urls', required=True)
parser.add_argument('-dataset', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    gen = DatasetGenerator(args.model, args.token, args.urls, args.dataset)
    gen.generate()
