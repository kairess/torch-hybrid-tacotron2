import korean_text.regexManager as regex

def baseball_domain_format(text, dictionary):
    domain_words = ["타율"]

    if any(word in text for word in domain_words):
        text = regex.baseball_domian_normalize_number(text, dictionary)
    #end if

    return text
#end def

def phone_number_format(text, dictionary):
    return regex.phone_number_normalize(text, dictionary.get_num_to_chinese_dic())
    

def account_format(text, dictionary):
    return regex.account_normalize_number(text, dictionary.get_num_to_chinese_dic())

"""
신용 및 대출 금일 만기
case 1) 00주 000원 (주, 원에 콤마 없이 전달)
"""
def amount_info_format_v1(text, dictionary):
    text = regex.detect_amount_info_v1(text, dictionary)
    text = text.replace('종목 $', '종목, $')
    num_units = dictionary.get_decimal_unit_list()[1:] + dictionary.get_ten_thousand_unit_list()[1:6]
    text = regex.amount_info_normalize(text, num_units)
    return text

"""
미수통보
case 1) 미수금 000원 (콤마 없이 전달)
case 2) 예정금액 000원 (콤마 없이 전달)
case 3) 담보부족액은 0000000원 (콤마 없이 전달)
"""
def amount_info_format_v2(text, dictionary):
    text = regex.detect_amount_info_v2(text, dictionary)
    num_units = dictionary.get_decimal_unit_list()[1:] + dictionary.get_ten_thousand_unit_list()[1:6]
    text = regex.amount_info_normalize(text, num_units)
    return text


"""
담보부족
case 1) 담보비율은 000% (% => 퍼센트 한글로 전달)
"""
def amount_info_format_v3(text, dictionary):
    text = regex.detect_amount_info_v3(text, dictionary)
    num_units = dictionary.get_decimal_unit_list()[1:] + dictionary.get_ten_thousand_unit_list()[1:6]
    text = regex.amount_info_normalize(text, num_units)
    return text

"""
투자
case 1) 000(원|유로|달러) 투자
"""
def amount_info_format_v4(text, dictionary):
    text = regex.detect_amount_info_v4(text, dictionary)
    num_units = dictionary.get_decimal_unit_list()[1:] + dictionary.get_ten_thousand_unit_list()[1:6]
    text = regex.amount_info_normalize(text, num_units)
    return text

if __name__ == '__main__':
    text = '대출만기 코스피종목 123주 1234원 연장불가입니다. 2~3배가 위험해요'
    from dicManager import DicManager
    dictionary = DicManager(None, None)
    print(amount_info_format_v1(text, dictionary))

    text = '금일 미수금 123원 예정금액 456원입니다. 1544-5000번 문의해주세요.'
    print(amount_info_format_v2(text, dictionary))

    text = '고객님 담보비율은 15.3 퍼센트이고 고객님 담보비율은 123.0 퍼센트이며, 마지막 담보비율은 1 퍼센트입니다. 그르치만 3~4배는 아직도 잘되죠, 점점 멀어지네요'
    print(amount_info_format_v3(text, dictionary))

    text = '고객님께서는 12312312원 투자하셨습니다. 또한 213123달러 투자하셨습니다 2413유로 투자한 것도 잊지 않으셨죠? 그르치만 3~4배는 아직도 잘되죠, 점점 멀어지네요'
    print(amount_info_format_v4(text, dictionary))
