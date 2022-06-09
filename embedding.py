labels = []
labels += ['bracelet', 'umbrella'] 
labels += ['background']
#//print(labels)
labels = [{'name': item, 'id': idx+1,} for idx, item in enumerate(labels)]
label_indices = {cat['id']: cat for cat in labels}
#//print(labels)



import clip
from tqdm import tqdm
import torch
import numpy as np

multiple_templates = [
    'There is {article} {} in the scene.', 'There is the {} in the scene.',
    'a photo of {article} {} in the scene.', 'a photo of the {} in the scene.', 'a photo of one {} in the scene.',
    # itap: I took a picture of
    'itap of {article} {}.', 'itap of my {}.', 'itap of the {}.',
    
    'a photo of {article} {}.', 'a photo of the {}.', 'a photo of one {}.', 'a photo of many {}.',
    'a good photo of {article} {}.', 'a good photo of the {}.',
    'a bad photo of {article} {}.', 'a bad photo of the {}.',
    'a photo of a nice {}.', 'a photo of the nice {}.',

    'a photo of a small {}.', 'a photo of the small {}.',
    'a photo of a large {}.', 'a photo of the large {}.',

    'a bright photo of {article} {}.', 'a bright photo of the {}.',
    'a dark photo of {article} {}.', 'a dark photo of the {}.',

    'a photo of a hard to see {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.', 'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.', 'a cropped photo of the {}.',
    'a close-up photo of {article} {}.', 'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.', 'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.', 'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.', 'a pixelated photo of the {}.',
    'a black and white photo of the {}.', 'a black and white photo of {article} {}.',

    'a painting of the {}.', 'a painting of a {}.',
]

clip.available_models()
# print(clip.available_models()) # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
model, preprocess = clip.load("ViT-B/32")



def article(name): 
  return 'an' if name[0] in 'aeiou' else 'a'

this_is = True
def build_text_embedding(categories):
  templates = multiple_templates
  with torch.no_grad():
    all_text_embeddings = []
    print('Building text embeddings...')
    for category in tqdm(categories):
      texts = [template.format(category['name'], article=article(category['name']))
        for template in templates]
      if this_is:
        texts = ['This is ' + text if text.startswith('a') or text.startswith('the') else text 
                 for text in texts]
      texts = clip.tokenize(texts) #@ Returns a LongTensor containing tokenized sequences of given text input(s).
      #TODO print("texts[tokenized sequences of given text inputs]: ", texts) ## This can be used as the input to the model.
      if torch.cuda.is_available():
        texts = texts.cuda()
        
      #@ Given a batch of text tokens, returns the text features encoded by the language portion of the CLIP model.  
      text_embeddings = model.encode_text(texts) #embed with text encoder 
      text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
      text_embedding = text_embeddings.mean(dim=0)
      text_embedding /= text_embedding.norm()
      all_text_embeddings.append(text_embedding)
    ## FOR ENDING
    all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
    if torch.cuda.is_available():
      all_text_embeddings = all_text_embeddings.cuda()
  ## WITH ENDING 
  return all_text_embeddings.cpu().numpy().T







text_features = build_text_embedding(labels)

for label, feature in zip(labels, text_features):
  label['embedding_feature'] = feature
  #//print(label)
  #//print(type(label))
#//print(text_features)
#//print(type(labels))

with open('label_embedding.csv', 'w') as f:
  for label in labels:
      [f.write('{0},{1}\n'.format(key, value)) for key, value in label.items()]
  
for feature in text_features:
  #//print(feature)
  pass

print(type(text_features))
np.save('label_embedding.npy', text_features)

features = np.load('label_embedding.npy')
import operator

print(operator.eq(features, text_features))
print(features)