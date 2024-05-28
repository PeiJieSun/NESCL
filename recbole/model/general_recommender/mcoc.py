# MCOc


# User-based collaborative filtering recommendation algorithms
# 1. calculate the user-user similarity
sparse_user_item_matrix = sparse(user_list, item_list) 
sparse_user_item_rating_matrix = sparse(user_list, item_list, rating) # (user, item, rating)
avg_rating = sparse_user_item_rating_matrix.mean(dim=1, keep_dims=True) # average rating for each user
sparse_user_item_rating_matrix = sparse_user_item_rating_matrix - (avg_rating * sparse_user_item_matrix) # (user, item, rating - user_avg_rating)

fenmu = (sparse_user_item_rating_matrix * sparse_user_item_rating_matrix).sum(dim=1, keep_dims=True) # average rating for each user
sim_uu = sp.matmul(sparse_user_item_rating_matrix, sparse_user_item_rating_matrix) / sp.sqrt(sp.matmul(fenmu, fenmu)) # user-user similarity

# recommend items based on the users' K-nearest neighbors. 
# select K-nearest neighbors for each user 
# matrix operation with user-item matrix
K_value = sim_uu[users] # get the value at rank-K
xx = (sim_uu > K_value) # batch * users_num
sp.matmul(xx, sparse_user_item_matrix) # batch * items_num
# rank the ratings of all items among all batch users, and recommend the top-K items to each user, remove the positive items of course.


# Item-based collaborative filtering recommendation algorithms
# 2. calculate the item-item similarity
item_avg_rating = sparse_user_item_rating_matrix.mean(dim=0, keep_dims=True) # average rating for each item
sparse_user_item_rating_matrix = sparse_user_item_rating_matrix - (item_avg_rating * sparse_user_item_matrix)

fenmu = (sparse_user_item_rating_matrix * sparse_user_item_rating_matrix).sum(dim=1, keep_dims=True) # average rating for each user
sim_ii = sp.matmul(sparse_user_item_rating_matrix, sparse_user_item_rating_matrix) / sp.sqrt(sp.matmul(fenmu, fenmu)) # user-user similarity

# recommend items based on the items' K-nearest neighbors. 
scores = (sparse_user_item_matrix[users], sim_ii) # batch * items_num

# 3. calculate the user-item similarity, i.e. user-item matrix


# implement fuzzy_c_means algorithm
def fuzzy_c_means():
    pass

