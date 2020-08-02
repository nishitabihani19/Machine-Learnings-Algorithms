import re
s='hi +913456789032 is a mo n. he ha +892345432345'
def mobile(s):
    
    phone_noRegex=re.compile(r'(\+\d\d)(\d\d\d\d\d\d\d\d\d\d)')
    mo=phone_noRegex.findall(s)
    print('number is',mo.group())
    if mo:
        return mo
    else:
        return None
    #print('number is',mo)
    
    
print(mobile(s))
