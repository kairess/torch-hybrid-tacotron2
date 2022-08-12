import korean_text.regexManager as regex
import korean_text.formatter as formatter


def format(text, dictionary):
    # text = formatter.baseball_domain_format(text, dictionary.get_decimal_point_unit_list())
    text = formatter.account_format(text, dictionary)
    text = formatter.phone_number_format(text, dictionary)
    # text = formatter.amount_info_format_v1(text, dictionary)
    # text = formatter.amount_info_format_v2(text, dictionary)
    # text = formatter.amount_info_format_v3(text, dictionary)
    # text = formatter.amount_info_format_v4(text, dictionary)
    # 영어 사전에 정의한 패턴 정규화
    if dictionary.get_eng_dic() is not None:
        text = regex.normalize_with_dictionary(text, dictionary.get_eng_dic())
    # end if
    return text
#end def

def clean(text):
    text = regex.collapse_whitespace(text)

    # 괄호 및 괄호 안 내용 제거
    text = regex.remove_words_in_parenthesis(text)

    # 한자 내용 제거
    text = regex.remove_chinese_words(text)

    # 일본어 내용 제거
    text = regex.remove_japanese_words(text)

    return text
#end def


def normalize(text, dictionary):
    # # 영어 사전에 정의한 패턴 정규화
    # if dictionary.get_eng_dic() is not None:
    #     text = regex.normalize_with_dictionary(text, dictionary.get_eng_dic())
    # # end if

    # term 사전에 정의한 패턴 정규화
    if dictionary.get_term_dic() is not None:
        text = regex.normalize_with_dictionary(text, dictionary.get_term_dic())
    # end if

    # 숫자뒤에 사용된 단위 정규화
    text = regex.normalize_number_with_unit(text, dictionary.get_unit_dic())

    # 두문자어로 판단되는 영어 단어 정규화 ex) 'Text To Speech' 의 두문자어는 'TTS' -> 티티에스
    text = regex.normalize_acronyms(text, dictionary.get_upper_to_kor_dic())

    # 숫자 정규화
    text = regex.normalize_number(text, dictionary)

    return text
#end def


if __name__ == '__main__':
    def test_norm(text):
        print(text)
        text = format(text)
        text = clean(text)
        text = normalize(text)
        print(text)
        print()
    # test_norm("저돌 (김한빈) 猪突 입니다.こととなり、、―の重要性など、った。 TTS korean 전처리 모듈 개발 중, SRT나 KTX를 타고 TTA 출장 JTBC에 방문 JTBCs로 가자")
    # test_norm("123USD는 -123,000KRW입니다. +128.235KRW입니다. 13마리 2018년 20가지 +0마리,-0년,0.0개, 0.0")
    # test_norm("99가지, 83가지 101가지 1153마리 ")
    # test_norm("10000년")
    # test_norm("20~30대")
    #
    test_norm("JTBC는  JTBCs를 DY는 A가 Absolute")
    test_norm("오늘(13일)  101마리 강아지가")
    test_norm('"저돌"(猪突) 입니다.~')
    test_norm('비대위원장이 지난 1월 이런 말을 했습니다. “난 그냥 산돼지처럼 돌파하는 스타일이다. 돌파하지 못하면 뚫는다. 알겠는가?”')
    test_norm("지금은 -12.35%였고 종류는 5가지와 19가지, 그리고 55가지였다")
    test_norm("JTBC는 TH와 K 양이 2017년 9월 12일 오후 12시에 40살이 된다")
    test_norm("이승엽 선수의 통산 타율은 0.543이며 평균 12.67개의 안타를 쳤다")
    
# Clean(일본어, 한자어 삭제 등등), normalization 2개로 분할
# formatted String ex) 타율 0.453 사할오푼삼리