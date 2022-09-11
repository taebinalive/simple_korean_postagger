kor_start = 44032
kor_end = 55203
cho_start = 12593
cho_end = 12622
jung_start = 12623
jung_end = 12643

cho_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ',
            'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ',
            'ㅌ', 'ㅍ', 'ㅎ']

jung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ',
             'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ',
             'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

jong_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
             'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
             'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
             'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def decompose(char):
    if char == '':
        return '', '', ''

    i = ord(char)

    # jaum only
    if cho_start <= i <= cho_end:
        return chr(i), '', ' '

    # moum only
    if jung_start <= i <= jung_end:
        return '', chr(i), ' '

    # not korean case
    if not kor_start <= i <= kor_end:
        return chr(i), '', ''

    i -= kor_start

    cho = i // 588
    jung = (i - cho * 588) // 28
    jong = i - cho * 588 - jung * 28

    return cho_list[cho], jung_list[jung], jong_list[jong]


def compose(cho, jung, jong):

    return chr((cho_list.index(cho) * 21 + jung_list.index(jung)) * 28 +
               jong_list.index(jong) + kor_start) if cho in cho_list else cho


if __name__ == "__main__":

    e = decompose('놃')
    print(compose(*e))
