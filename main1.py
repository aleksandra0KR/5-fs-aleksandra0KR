import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from BuildIn import built_in_mine
from Filter import filter_mine
from LibraryRealizationSelection import filter_method, embedded_method, wrapper_method
from Wrapper import wrapper_mine

data = open('SMS.tsv', encoding='UTF-8')
file_lines = data.readlines()
labels, raw_inputs = list(zip(*[tuple(file_line.split(maxsplit=1)) for file_line in file_lines]))

vector = CountVectorizer(stop_words='english')
count_vector = vector.fit_transform(list(raw_inputs))
vector_df = pd.DataFrame(count_vector.toarray(), columns=vector.get_feature_names_out())

label_mapper = {
    'spam': 0,
    'ham': 1
}
y = np.array([label_mapper[label] for label in labels])
X = vector_df.to_numpy()
feature_names = vector_df.columns.to_numpy()

selected_features_built_in = built_in_mine(X, y, feature_names)
print("Selected features using built_in method:", selected_features_built_in)
selected_features_built_in_library = embedded_method(X, y, feature_names)
print("Selected features using built_in method:", selected_features_built_in_library)
'''
Selected features using built_in method: ['txt' 'claim' 'bak' 'think' 'free' 'www' 'evening' 'text' 'vikky'
 'finished' 'urgent' 'accounts' 'mins' 'working' 'oh' '150p' 'lt' 'cann'
 'stop' 'blake' 'pete' '50' 'win' 'pls' 'dino' 'having' 'service' 'video'
 'shd' 'days']
'''
'''
Selected features using built_in method: ['100' '1000' '150p' '16' '18' '50' '500' 'cash' 'chat' 'claim' 'code'
 'com' 'cs' 'free' 'mobile' 'nokia' 'prize' 'reply' 'ringtone' 'service'
 'stop' 'text' 'tone' 'txt' 'uk' 'urgent' 'video' 'win' 'won' 'www']
'''

selected_features_filer = filter_mine(X, y, feature_names)
print("Selected features using filter method:", selected_features_filer)
selected_features_filer_library = filter_method(X, y, feature_names)
print("Selected features using filter method library:", selected_features_filer_library)
'''
Selected features using filter method: ['txt' 'free' 'claim' 'www' 'mobile' 'prize' '150p' 'stop' 'uk' 'text'
 'won' 'reply' 'urgent' '16' '18' 'guaranteed' 'cash' 'service' '50' 'win'
 '500' 'cs' 'contact' 'nokia' '1000' 'customer' '100' 'awarded' 'com'
 'tone']
'''
'''
Selected features using filter method library: ['100' '1000' '150p' '16' '18' '50' '500' 'awarded' 'cash' 'claim' 'com'
 'contact' 'cs' 'customer' 'free' 'guaranteed' 'mobile' 'nokia' 'prize'
 'reply' 'service' 'stop' 'text' 'tone' 'txt' 'uk' 'urgent' 'win' 'won'
 'www']
'''

selected_features_wrapper = wrapper_mine(X, y, feature_names)
print("Selected features using wrapper method:", selected_features_wrapper)
'''
Selected features using wrapper method: ['gt' '00' '000' '000pes' '008704050406' '0089' 'www' 'video' 'liked'
 'good' 'hi' 'bak' 'bakrid' 'later' 'beverage' 'finish' 'txt' 'service'
 'road' 'type' '800' 'claim' 'barmed' '150p' 'id' 'great' '50' '16' 'did'
 'mobile']
'''
selected_features_wrapper_library = wrapper_method(X, y, feature_names)
print("Selected features using wrapper method library:", selected_features_wrapper_library)

'''
Selected features using wrapper method library: ['gt' '800' '000' 'did' '0089' '150p' 'liked' 'service'
 'good' 'hi' '000pes' 'bak' 'bakrid' 'later' 'www' 'beverage' 'finish' 'txt' 'barmed'
 'mobile' 'road' 'type' '00' 'claim' 'video' 'id' 'great' '50' '16' 
 '008704050406']
'''
