import re, ast
import korean_text.regexPatternConst as reConst


# 이중 공백 제거
def collapse_whitespace(text):
    pattern_const = reConst.white_space
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return re.sub(pattern, ' ', text)


# 괄호 안 내용 삭제
def remove_words_in_parenthesis(text):
    pattern_const = reConst.words_in_parenthesis
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return " ".join(re.sub(pattern, '', text).split())


# end def

# 한자어 삭제
# 한자어 마침표(고리점), 쉼표(모점), 줄표 제거 ([。|、|―]+)
# http://contents.kocw.or.kr/document/lec/2011_2/hufs/04/Chinese_01_01.pdf
def remove_chinese_words(text):
    # all chinese
    chinese = reConst.all_chinese
    token = reConst.chinese_token
    pattern = re.compile(u'({chinese})|({token})'.format(chinese=chinese, token=token), re.UNICODE)
    return " ".join(re.sub(pattern, '', text).split())


# end def

# 일본어 삭제
# 3000-303F : punctuation
# 3040-309F : hiragana
# 30A0-30FF : katakana
# FF00-FFEF : Full-width roman + half-width katakana
# 4E00-9FAF : Common and uncommon kanji
def remove_japanese_words(text):
    # all japanese
    japanese = reConst.all_japanese
    pattern = re.compile(u'{japanese}'.format(japanese=japanese), re.UNICODE)
    return " ".join(re.sub(pattern, '', text).split())


# end def


# # 한자어 마침표(고리점), 쉼표(모점), 줄표 제거
# # http://contents.kocw.or.kr/document/lec/2011_2/hufs/04/Chinese_01_01.pdf
# def __remove_chinese_tokens(text):
#     # chinese token
#     pattern = re.compile(u'[。|、|―]+', re.UNICODE)
#     return " ".join(re.sub(pattern, '', text).split())
# #end def

# 사전 정의 단어 정규화
def normalize_with_dictionary(text, dictionary):
    def replace(match):
        key = match.group()
        if key in dictionary:
            return dictionary.get(key)
        else:
            return key
        # end if

    # end def

    sorted_keys = sorted([key for key in dictionary.keys()], key=len, reverse=True)
    if any(key in text for key in sorted_keys):
        pattern = re.compile(u'|'.join(re.escape(key) for key in sorted_keys), re.UNICODE)
        return re.sub(pattern, replace, text)
    else:
        return text
    # end if
# end def

# 두문자어 정규화
def normalize_acronyms(text, dictionary):
    def replace(match):
        acronym = match.group()
        if all([char.isupper() for char in acronym]):
            return "".join(dictionary[char] for char in acronym)
        else:
            return acronym
        # end if

    # end def

    pattern_const = reConst.all_english
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return re.sub(pattern, replace, text)


# end def

# 숫자 뒤 단위 정규화
def normalize_number_with_unit(text, dictionary):
    number_pattern = reConst.number_with_symbol
    # 단어길이가 긴 단위부터 정규표현식에 잡히도록 key 정렬 ex) (mil|mile)와 같이 길이가 짧은 단위가 앞에오면 100mile --> 100밀e 로 바뀌는 것 방지
    sorted_keys = sorted([key for key in dictionary.keys()], key=len, reverse=True)
    unit_pattern = "|".join(re.escape(unit) for unit in sorted_keys)

    def replace(match):
        number = match.group(1)
        unit = match.group(3)
        return "{number}{unit}".format(number=number, unit=dictionary.get(unit))

    # end def

    if any(key in text for key in dictionary.keys()):
        pattern = re.compile(u'({number})({unit})'.format(number=number_pattern, unit=unit_pattern), re.UNICODE)
        return re.sub(pattern, replace, text)
    else:
        return text
    # end if


# end def

# 숫자 정규화
def normalize_number(text, dicManager):
    number_pattern = reConst.number_with_symbol
    sorted_native_unit_list = sorted(dicManager.get_native_unit_list(), key=len, reverse=True)
    unit_pattern = "|".join(re.escape(unit) for unit in sorted_native_unit_list)

    pattern = re.compile(u'({number})({unit})*'.format(number=number_pattern, unit=unit_pattern), re.UNICODE)

    text = re.sub(pattern, _normalize_sign_and_zero, text)

    text = re.sub(pattern, lambda match: _normalize_number_to_kor(match, dicManager), text)

    return text


# end def

def _normalize_number_to_kor(match, dicManager):
    num_str, unit_str = match.group(1), match.group(3) if not match.group(3) is None else ''
    has_native_unit = True if not unit_str == '' else False

    num = ast.literal_eval(num_str)

    num_split = num_str.split(".")

    digit_str, float_str = num_split[0], num_split[1] if len(num_split) == 2 else ''

    thousand_rule_list = dicManager.get_ten_thousand_unit_list()
    decimal_list = dicManager.get_decimal_unit_list()
    native_decimal_dic = dicManager.get_native_decimal_dic()
    num_to_native_dic = dicManager.get_num_to_native_dic()
    num_to_chinese_dic = dicManager.get_num_to_chinese_dic()

    digit_to_kor = ""
    if has_native_unit and int(digit_str) < 100 and float_str == '':
        digit_to_kor = _number_to_kor(digit_str, num_to_native_dic, decimal_list, thousand_rule_list)
        digit_to_kor = normalize_with_dictionary(digit_to_kor, native_decimal_dic)
    else:
        digit_to_kor = _number_to_kor(digit_str, num_to_chinese_dic, decimal_list, thousand_rule_list)
        # 2021/01/08 국립국어원 표준에 맞게 변경 --> 5012가지 --> 오천십이가지
        # if has_native_unit and float_str == '':
        #     digit_to_kor = _normalize_first_digit(digit_to_kor, num_to_chinese_dic, num_to_native_dic)

        # end if
    # end if

    float_to_kor = ''
    if not float_str == '':
        float_kor = normalize_with_dictionary(float_str, num_to_chinese_dic)
        float_to_kor = "점{kor}".format(kor=float_kor)
    # end if

    kor = "{digit}{float}{unit}".format(digit=digit_to_kor, float=float_to_kor, unit=unit_str)

    return kor


# end def

def _normalize_first_digit(text, chinese_dic, native_dic):
    def mkDic(chinese, native):
        dic = {}
        for chinese_key in chinese.keys():
            for native_key in native.keys():
                if chinese_key == native_key:
                    dic[chinese[chinese_key]] = native[native_key]
                # end if
            # end for
        # end for
        return dic

    # end def
    first_digit = text[-1]
    if first_digit in chinese_dic.values():
        chinese_to_native = mkDic(chinese_dic, native_dic)
        replace_digit = chinese_to_native[first_digit]
        return "{prefix}{suffix}".format(prefix=text[:-1], suffix=replace_digit)
    else:
        return text
    # end if


# end def

# def _kor_to_native_pronunce(text, native_decimal_dic):
#     if any(word in text for word in native_decimal_dic):
#         pronunce_pattern = "|".join(re.escape(pronunce) for pronunce in native_decimal_dic.keys())
#         pattern = re.compile(u'{pronunces}'.format(pronunces=pronunce_pattern), re.UNICODE)
#         num_to_kor = re.sub(pattern, replace, text)
# #end def

def _number_to_kor(number, num_to_kor_dic, decimal_rule_dic, thousand_rule_dic):
    if number == '0':
        return '영'

    num_size = len(number)
    temp_kor = []
    kor = ""
    for i, key in enumerate(number, start=1):
        num = int(key)
        index = num_size - i
        decimal_rule = index % 4
        thousand_rule = int(index / 4)
        if num != 0:
            kor_value = num_to_kor_dic[key]
            decimal_value = decimal_rule_dic[decimal_rule]
            if kor_value == num_to_kor_dic["1"] and not decimal_rule == 0:
                temp_kor.append(decimal_value)
            else:
                temp_kor.append(kor_value)
                temp_kor.append(decimal_value)
            # end if
        # end if

        if decimal_rule == 0 and len(temp_kor) != 0:
            append_kor = "".join(temp_kor)
            kor = "{prev_kor}{append_kor}".format(prev_kor=kor, append_kor=append_kor)
            temp_kor = []
            kor = "{kor}{thousand_rule}".format(kor=kor, thousand_rule=thousand_rule_dic[thousand_rule])
        # end if
    # end for

    if kor.startswith(num_to_kor_dic["1"]) and not kor == num_to_kor_dic["1"]:
        kor = kor[1:]
        for tens_unit in thousand_rule_dic[2:]:
            if kor.startswith(tens_unit):
                kor = "일" + kor
                break
            # end if
        # end for
    # end if

    return kor


# end def

def _normalize_sign_and_zero(match):
    num_str, unit_str = match.group(1), match.group(3) if not match.group(3) is None else ''
    sign_str = ''

    if num_str.startswith("+"):
        sign_str = "플러스 "
        num_str = num_str[1:]
    elif num_str.startswith("-"):
        sign_str = "마이너스 "
        num_str = num_str[1:]
    # end if

    num_str = num_str.replace(',', '')

    try:
        num = ast.literal_eval(num_str)
        if num < 1:
            num = num_str

    except Exception as e:
        if num_str.startswith('0') and not '.' in num_str:
            num = int(num_str)
        else:
            raise e

    if num == 0:
        return "{sign}{num}{unit}".format(sign=sign_str, num="영", unit=unit_str)
    # end if

    return "{sign}{num}{unit}".format(sign=sign_str, num=num, unit=unit_str)


# end def

def baseball_domian_normalize_number(text, dictionary):
    def replace(match):
        number = match.group(2)
        replace_number = ""
        for i, digit in enumerate(number):
            unit = dictionary[int(i)]
            replace_number += "{digit}{unit}".format(digit=digit, unit=unit)
        # end for

        return replace_number

    # end def

    pattern_const = reConst.decimal_point_number
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return re.sub(pattern, replace, text)


# end def

def phone_number_normalize(text, num_to_kor):
    def replace(match):
        phone_number = match.group()
        units = [unit for unit in phone_number]

        norm_units = []
        for unit in units:
            if unit != '-':
                norm_units.append(num_to_kor[unit])
            else:
                norm_units.append(' ')

        phone_number = ''.join(norm_units)
        return phone_number.replace('영', '공')

    pattern_const = reConst.phone_number
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return re.sub(pattern, replace, text)


def account_normalize_number(text, num_to_kor):
    def replace(match):
        account = match.group()
        account = account.replace('^', '')
        units = [unit for unit in account]

        norm_units = []
        for unit in units:
            norm_units.append(num_to_kor[unit])

        account = ', '.join(norm_units)
        return account

    pattern_const = reConst.account
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return re.sub(pattern, replace, text)


def detect_amount_info_v1(text, dictionary):
    def replace(match):
        amount_info = match.group()
        amount_info = normalize_number(amount_info, dictionary)
        amount_info = '${}$'.format(amount_info)
        return amount_info

    pattern_const = reConst.amount_info_v1
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return re.sub(pattern, replace, text)


def detect_amount_info_v2(text, dictionary):
    def replace(match):
        amount_info = match.group()
        amount_info = normalize_number(amount_info, dictionary)
        amount_info = '${}$'.format(amount_info)
        amount_info = amount_info.replace(match.group(2), match.group(2) + ',')
        return amount_info

    pattern_const = reConst.amount_info_v2
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return re.sub(pattern, replace, text)


def detect_amount_info_v3(text, dictionary):
    def replace(match):
        amount_info = match.group()
        amount_info = normalize_number(amount_info, dictionary)
        amount_info = '${}$'.format(amount_info)
        amount_info = amount_info.replace(match.group(2), match.group(2) + ',').replace('점', ' 쩜 ')
        return amount_info

    pattern_const = reConst.amount_info_v3
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return re.sub(pattern, replace, text)


def detect_amount_info_v4(text, dictionary):
    def replace(match):
        amount_info = match.group()
        amount_info = normalize_number(amount_info, dictionary)
        amount_info = '${}$'.format(amount_info)
        return amount_info

    pattern_const = reConst.amount_info_v4
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return re.sub(pattern, replace, text)


def amount_info_normalize(text, num_units):
    def replace(match):
        amount_info = match.group()
        amount_info = amount_info.replace('$', '')

        norm_amount_info = []

        comma_units = reConst.amount_info_v4_units + ['주']

        for ch in amount_info:
            norm_amount_info.append(ch)
            if ch in num_units:
                norm_amount_info.append(' ')

            # if ch == '주' or ch == '원':
            #     norm_amount_info.append(',')

            if any([''.join(norm_amount_info).endswith(unit) for unit in comma_units]):
                norm_amount_info.append(',')

        return ''.join(norm_amount_info)

    pattern_const = reConst.detect_amount_info
    pattern = re.compile(u'{const}'.format(const=pattern_const), re.UNICODE)
    return re.sub(pattern, replace, text)
