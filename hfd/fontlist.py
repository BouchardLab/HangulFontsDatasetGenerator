import os
import numpy as np



fontname_folder = os.path.join(os.environ['HOME'],'data/hangul/h5s')
fontnames = os.listdir(fontname_folder)
fontnames = [f for f in fontnames if ('ttf' in f.lower() or 'otf' in f.lower())]

hmc1_fontnames = os.listdir(fontname_folder)
hmc1_fontnames = [f for f in hmc1_fontnames if ('ttf' in f.lower() or 'otf' in f.lower())]

fontsizes = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
             30, 36, 42, 48, 54, 60, 66, 72]

fonts_with_imf = ['SourceHanSerifK-Regular', 'NanumGothicExtraBold',
                  'NanumGothicBold', 'NanumGothic', 'GothicA1-Black',
                  'GothicA1-Bold', 'GothicA1-ExtraBold', 'GothicA1-ExtraLight',
                  'GothicA1-Light', 'GothicA1-Medium', 'GothicA1-Regular',
                  'GothicA1-SemiBold', 'GothicA1-Thin']
