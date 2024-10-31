from text import chinese, cleaned_text_to_sequence, symbols, english
from text.minbpe.minbpe import GPT4Tokenizer

language_module_map = {
    'zh': chinese,
    'en': english
}
special = [
    ('%', 'zh', "SP"),
    ('￥', 'zh', "SP2"),
    ('^', 'zh', "SP3"),
]
def clean_text(text, language):
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    if(language=="zh"):
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    else:
        phones = language_module.g2p(norm_text)
        word2ph=None

    for ph in phones:
        assert ph in symbols
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    """
    Special mute segment sp symbol handling
    """
    text = text.replace(special_s, ",")
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones:
        assert ph in symbols
        if ph == ',':
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph

def text_to_sequence(text, language):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)

tokenizer = GPT4Tokenizer()
def clean_text_BPE(text, language):

    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)

    phone_ids = tokenizer.encode(norm_text, allowed_special="all")
    word2ph = None

    return phone_ids, word2ph, norm_text


if __name__ == '__main__':
    print(clean_text_BPE("测试", 'zh'))


