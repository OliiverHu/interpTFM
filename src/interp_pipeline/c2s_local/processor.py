"""
C2SProcessor: data processing pipeline for C2S-Scale models.

Handles everything between raw AnnData and tokenizer-ready strings:
    normalize → build vocab → generate cell sentences
"""

from collections import OrderedDict
from typing import List, Optional

from datasets import Dataset

from .util import generate_vocabulary, generate_sentences
from .prompt_formatter import C2SPromptFormatter

class C2SProcessor:
    """
    Converts AnnData objects into tokenizer-ready cell sentence strings.

    Follows a fit/transform pattern so the vocabulary is built once on the
    full dataset and reused when transforming individual batches:

        processor = C2SProcessor(max_genes=2048)
        processor.fit(full_adata)

        for batch_adata in batches:
            sentences = processor.transform(batch_adata)
            encoded   = tokenizer(sentences, max_length=512)
    """

    def __init__(
        self,
        vocab: Optional[OrderedDict] = None,
        max_genes: Optional[int] = None,
    ):
        """
        Args:
            vocab: Pre-built vocabulary (gene → non-zero cell count).
                   If None, call fit() before transform().
            max_genes: Truncate each cell sentence to this many genes.
        """
        self.vocab = vocab
        self.max_genes = max_genes

    def normalize_adata(
        self,
        adata,
        min_genes: int = 200,
        min_cells: int = 3,
        base: int = 10,
    ):
        """
        Normalize AnnData for C2S-Scale input.

        Args:
            adata: AnnData object to normalize (modified in-place)

        Returns:
            The normalized AnnData object
        """
        import scanpy as sc
        if min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)
        if min_cells > 0:
            sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata, base=base)
        return adata

    def adata_to_arrow(self,
            adata,
            sentence_delimiter: str = ' ',
            label_col_names: list = ["cell_type", "organism"], 
    ):
        """
        Build the gene vocabulary from a (full) AnnData object.

        Should be called once on the complete dataset so that gene ordering
        is consistent across batches.

        Returns:
            self, to allow chaining: processor.fit(adata).transform(batch)
        """
        first_gene_name = str(adata.var_names[0])
        if "ENS" in first_gene_name:
            print(
                """WARN: adata.var_names seems to contain ensembl IDs rather than gene/feature names. 
                It is highly recommended to use gene names in cell sentences."""
            )
        
        # Create vocabulary and cell sentences based on adata object
        vocabulary = generate_vocabulary(adata)
        sentences = generate_sentences(adata, vocabulary, delimiter=sentence_delimiter)
        cell_names = adata.obs_names.tolist()

        # Build arrow dataset dict
        data_dict = {
            "cell_name": cell_names,
            "cell_sentence": sentences,
        }
        if label_col_names is not None:
            for col in label_col_names:
                data_dict[col] = adata.obs[col].tolist()

        arrow_dataset = Dataset.from_dict(data_dict)
        return arrow_dataset, vocabulary

    def prompts_generation(
        self,
        arrow_dataset,
        task: str = "cell_type_prediction",
        n_genes: int = 200,
    ):
        """
        Format an arrow dataset (output of adata_to_arrow) into cell type prediction prompts.

        Args:
            arrow_dataset: Dataset returned by adata_to_arrow.
            task: The task type for prompt formatting.
            n_genes: Number of top genes to include per cell sentence.

        Returns:
            Formatted Huggingface Dataset with 'model_input', 'response', and 'sample_type' columns.
        """

        prompt_formatter = C2SPromptFormatter(task=task, top_k_genes=n_genes)
        return prompt_formatter.format_hf_ds(arrow_dataset)


    