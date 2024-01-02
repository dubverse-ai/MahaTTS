import sys
from maha_tts.config import config

labels=" abcdefghijklmnopqrstuvwxyz.,:;'()?!\""
labels=" !\"'(),-.:;?[]abcdefghijklmnopqrstuvwxyzàâèéêü’“”"
labels='''ଊతూിਮ০य़లഢਪਟକఝૂएड‌`यঢअచଢ଼ਧ—ତলશರଖच,பવड़ષंಈಮਤਇଥkखഗబ= इਸಣਹછ™ୟ.ोೀৎುഊଳંർਘମഴఙसଗൃlଝਜఇഓਐভയಅಠభാടਔಒ೧পஜaૅૠএଲ৯eകँ৭àৱऊટഒਗহિేயీെஈଓഭೊাੌಙ१ଈःസठખm‘ొऍಿcശrట।ऱଋઘਛெਬಂङಹஞ਼ભ১"એੂചಸગಷ়ଁമಓtஒઉಪs్-pଛ›ढ+ಆ'বનধৰউીଅઝ੍ೂʼൂఔfતषഖঢ়৬﻿ਖक़ਵషணझപળଔઞੇವௗઁത২xెഥख़iটਲધಔೇீથ*ഝॅঃஓूఒীనਜ਼எુுహौ९ൗౌফഔોhஔণంफ़ఋçଯઊൽଆ’ୁைഛ२&ঁണ़ైৌআஆোਠਭजொમळಘஷഏি/ચਾ“ਯ$ଐീवऩ८ઢఛఎেథഠ[औಳରथୃൈಝnজਥऑଷੱल೯wओଵढ़மവरడఊbೖਈૃपdêଉఐ;ै ఢ	ઔકচ৩‎ਊൾഉਕ೦ಏj€:ਦಗાളੁशफുழൻಊगફఏఅ?णറഘಞ४ಡಫଠ್ড೨ൊঞमਂસૉॉઅരஙલঘନ്ఠॄvઋృষऎகೕଘઆఞലେূஊఉૈദఫఈदকज़!ధઠవଞறಟਖ਼ਫ਼ইਢഡঠஃஸୂटঅହఆளోईৃಜ॥(ઈଏੀഈक્গ ಚಢഹೃिஏಯyশேଡೋੈਣડఃഷഇਸ਼நখಋோনૐਏgहৗೈृவੰଜग़ੋ୍)ൌరమൺংञਓપయധஇോ५ઃಲళঊತॽ­ന…ঙಭाಇउਅଶরઓି্ூমuపബ\ૌଟबਆुಕଫதছ३దਿದణஐௌ்ৈqఘலહಾ०ಛঐிওऋ‍ి৮ेਨଇүଧഞಶéਚ्৫ୋశఓદঈୀ৪ପüুങਗ਼ઑજথఖঝಐऽਰାആജीઇੜ]आବଡ଼ഫಥుಎણଃયछஅેஹംଢબoদഎగଭాേഅঋসഐಃzਡಬਝன–உಖಉഃযସୈೆకॐನഋয়సசଙড়ୱऒऐઐतଂாতરâèनಧ॑டঔभர”​జ৷ਫଣଚଦधघೌୌਉ'''

labels= [i for i in labels]

text_labels = [i for i in labels]
text_labels+='<S>','<E>','<PAD>'

code_labels= [str(i) for i in range(config.semantic_model_centroids)]
labels+=code_labels
code_labels+='<SST>','<EST>','<PAD>'

labels+='<S>','<E>','<SST>','<EST>','<PAD>'

tok_enc = {j:i for i,j in enumerate(labels)}
tok_dec = {i:j for i,j in enumerate(labels)}

#text encdec
text_enc = {j:i for i,j in enumerate(text_labels)}
text_dec = {i:j for i,j in enumerate(text_labels)}

#code encdec
code_enc = {j:i for i,j in enumerate(code_labels)}
code_dec = {i:j for i,j in enumerate(code_labels)}

# print('length of the labels: ',len(labels))