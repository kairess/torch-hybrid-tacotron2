import os
import json

_root_path = os.path.dirname(os.path.realpath(__file__))
_comment = '#'
_dictionaries_dir = os.path.join(_root_path, "dic")
_config_path = os.path.join(_root_path, "property", "config.json")

class DicManager:
    def __init__(self, eng_dic, term_dic):
        self._config = _get_config(_config_path)
        self._eng_dic = eng_dic
        self._term_dic = term_dic
        self._upper_to_kor_dic = _get_dictionary(os.path.join(_dictionaries_dir, self._config.get('upper_to_kor_dic')))
        self._num_to_chinese_dic = _get_dictionary(os.path.join(_dictionaries_dir, self._config.get('num_to_chinese_dic')))
        self._num_to_native_dic = _get_dictionary(os.path.join(_dictionaries_dir, self._config.get('num_to_native_dic')))
        self._unit_dic = _get_dictionary(os.path.join(_dictionaries_dir, self._config.get('unit_dic')))
        self._native_unit_list =_get_list(os.path.join(_dictionaries_dir, self._config.get('native_unit_list')))
        self._native_decimal_dic = {"십": "열", "두십": "스물", "세십": "서른", "네십": "마흔", "다섯십": "쉰", "여섯십": "예순", "일곱십": "일흔", "여덟십": "여든", "아홉십": "아흔"}
        self._decimal_unit_list = [""] + list("십백천")
        self._decimal_point_unit_list = list("할푼리모사홀미섬사진애묘막") + ["모호", "준순", "수유", "순식", "탄지", "찰나" ,"육덕", "허공", "청정"]
        self._ten_thousand_unit_list = [""] + list("만억조경해자양구간정재극") + ["항아사", "아승기", "나유타", "불가사의", "무량대수"]
    #end def

    def get_dictionaries(self):
        return [dic for dic in self._config.keys()]
    #end def

    def get_eng_dic(self):
        return self._eng_dic
    #end def

    def get_term_dic(self):
        return self._term_dic
    #end def

    def get_upper_to_kor_dic(self):
        return self._upper_to_kor_dic
    #end def

    def get_num_to_chinese_dic(self):
        return self._num_to_chinese_dic
    #end def

    def get_num_to_native_dic(self):
        return self._num_to_native_dic
    #end def

    def get_native_decimal_dic(self):
        return self._native_decimal_dic
    #end def

    def get_decimal_unit_list(self):
        return self._decimal_unit_list
    #end def

    def get_decimal_point_unit_list(self):
        return self._decimal_point_unit_list
    #end def

    def get_ten_thousand_unit_list(self):
        return self._ten_thousand_unit_list
    #end def

    def get_native_unit_list(self):
        return self._native_unit_list
    # end def

    def get_unit_dic(self):
        return self._unit_dic
    # end def

#end class

def _get_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            config = json.load(f)
        except Exception as e:
            config = {}
    # end with
    return config
#end def

def _get_dictionary(path):
    dic = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace('_', ' ')
            if not line.startswith(_comment) and len(line) > 0:
                token = line.split("\t")
                key = token[0]
                value = token[1]
                dic[key] = value
            #end if
        #end for
    return dic
#end def

def _get_list(path):
    list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line.startswith(_comment) and len(line) > 0:
                list.append(line)
            #end if
        #end for
    return list
#end def

if __name__ == '__main__':
    dic = DicManager(None, None)
    print(dic.get_dictionaries())
    print(dic.get_eng_dic())
    print(dic.get_upper_to_kor_dic())
    print(dic.get_num_to_chinese_dic())
    print(dic.get_num_to_native_dic())
    print(dic.get_decimal_unit_list())
    print(dic.get_ten_thousand_unit_list())
    print(dic.get_native_unit_list())