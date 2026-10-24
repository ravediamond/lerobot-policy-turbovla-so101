"""Tiny local DINOv3 and BERT model directories, so tests run offline (no Hub, no gated weights).

Usable from both the plugin environment and upstream's reference environment (only `transformers`
is needed). The tokenizer is a real `BertTokenizerFast` on a small vocabulary, so upstream's
Grounding-DINO sub-sentence masks (split on [CLS] / [SEP] / "." / "?") behave as with BERT-base.
"""

from pathlib import Path

IMAGE_SIZE = 64
PATCH_SIZE = 16
VISION_WIDTH = 32
LANGUAGE_WIDTH = 24

WORDS = (
    "put the blue bowl in pink pick up block number red cube on top of a push it to left right side "
    "open drawer close stack cups move card away grab roller press stapler 0 1 2 3 4 5 6 7"
).split()


def make_backbone_dirs(root: str | Path) -> tuple[str, str]:
    from transformers import BertConfig, BertTokenizerFast, DINOv3ViTConfig

    root = Path(root)
    vision_dir, text_dir = root / "tiny-dinov3", root / "tiny-bert"
    vision_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    DINOv3ViTConfig(
        hidden_size=VISION_WIDTH,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=2 * VISION_WIDTH,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
    ).save_pretrained(vision_dir)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "?", ",", *WORDS]
    (text_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
    # A `tokenizer.json` built with `tokenizers` directly, as bert-base-uncased ships one: transformers
    # 4.x and 5.x both load it as is (5.x no longer builds a fast tokenizer from `vocab_file=`).
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    ids = {w: i for i, w in enumerate(vocab)}
    tok = Tokenizer(models.WordPiece(vocab=ids, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", ids["[CLS]"]), ("[SEP]", ids["[SEP]"])],
    )
    BertTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        do_lower_case=True,
    ).save_pretrained(text_dir)
    BertConfig(
        vocab_size=len(vocab),
        hidden_size=LANGUAGE_WIDTH,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=2 * LANGUAGE_WIDTH,
        max_position_embeddings=64,
    ).save_pretrained(text_dir)
    return str(vision_dir), str(text_dir)
