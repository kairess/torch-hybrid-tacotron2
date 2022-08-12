
# 괄호안 모든 content
words_in_parenthesis = "\([^)]*\)"

# 공백
white_space = "\s+"

# 모든 중국어
all_chinese = "[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]+"

# 중국어 토큰
chinese_token = "[。|、|―]+"

# 모든 일본어
all_japanese = "[\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\u4e00-\u9faf]+"

# 모든 영어
all_english = "[A-Za-z]+"

# 부호가 포함된 숫자 패턴 ex) 12, +12, -12, 12.54, +12.54
number_with_symbol = "([+-]?\d[\d,]*)[\.]?\d*"

# 소수 패턴
decimal_point_number = "(0\.)(\d+)"

# phone number 패턴
phone_number = '([0-9]{2,3}-[0-9]{3,4}-[0-9]{4})'

# formatter Account 패턴
account = "(\^[0-9]+\^)"

# amount info 패턴
amount_info_v1 = "([0-9]+주 [0-9]+원)"
amount_info_v2 = "((미수금|예정금액|담보부족액은) [0-9]+원)"
amount_info_v3 = "((담보비율은) [0-9]*[\.]?[0-9]* 퍼센트)"
amount_info_v4_units = ['원', '달러', '유로', '엔화', '위안']
amount_info_v4 = "([0-9]+({}) 투자)".format('|'.join(amount_info_v4_units))
detect_amount_info = "(\$[가-힣 ,]+\$)"