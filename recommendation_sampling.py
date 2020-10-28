import numpy as np
import scipy.sparse

items = np.load('/Users/mattmerrill/Springboard/Capstone2/olist_datascience'
                '/exploration/Docker_files/items.npy', allow_pickle=True)
                
user_to_product_interaction = scipy.sparse.load_npz('/Users/mattmerrill/Springboard/Capstone2/olist_datascience'
                      								'/exploration/Docker_files/user_to_product_interaction.npz')

user_to_feature_interaction = scipy.sparse.load_npz('/Users/mattmerrill/Springboard/Capstone2/olist_datascience'
                      								'/exploration/Docker_files/user_to_feature_interaction.npz')
                     								
user_to_index_mapping = np.load("/Users/mattmerrill/Springboard/Capstone2/olist_datascience"
                      			"/exploration/Docker_files/user_to_index_mapping.pkl", allow_pickle=True)

class recommendation_sampling:
    
	def __init__(self, model, items = items, user_to_product_interaction_matrix = user_to_product_interaction, 
                user2index_map = user_to_index_mapping):
        
		self.user_to_product_interaction_matrix = user_to_product_interaction_matrix
		self.model = model
		self.items = items
		self.user2index_map = user2index_map
    
	def recommendation_for_user(self, user, user_features=None):
        
		# getting the userindex
        
		userindex = self.user2index_map.get(user, None)
        
		if userindex == None:
			return None
        
		users = [userindex]
        
		# products already bought
        
		known_positives = self.items[self.user_to_product_interaction_matrix.tocsr()[userindex].indices]
        
		# scores from model prediction
		scores = self.model.predict(user_ids = users, item_ids = np.arange(self.user_to_product_interaction_matrix.shape[1]),
                                    user_features=user_features,
                                    item_features = product_to_feature_interaction)

		# top items
        
		top_items = self.items[np.argsort(-scores)]
        
		# printing out the result
		print("User %s" % user)
		print("     Known positives:")
        
		for x in known_positives[:3]:
			print("                  %s" % x)
			print("                  {}".format(product_to_feature['feature'][product_to_feature['product_id'] == x].iloc[0]))
            
            
		print("     Recommended:")
        
		for x in top_items[:3]:
			print("                  %s" % x)
			print("                  {}".format(product_to_feature['feature'][product_to_feature['product_id'] == x].iloc[0]))