<h1>API</h1>

1. <b>GET / -
initialize the RecSys (always send this as the first request).

2. <b>GET /train</b> - retrain the model if needed (for example, if a lot of new
mashups have arrived to the database).

3. <b>GET /recommend?id=<i>666</i></b> -
get the recommendations list in form of json containing mashup ids
for the provided user id. Optionally, you may provide custom values for
any of the following parameters
(it will not override the default values shown below, 
i.e. the values provided apply only for the current request):
    ```
    {
                'liked_pop_size': 3,
                'most_listened_pop_size': 3,
                'recently_listened_pop_size': 3,
                'base_l_neighb': 5,
                'base_m_neighb': 5,
                'base_r_neighb': 5,
                'playlist_neighb': 3,
                'playlist_cand_lim': 5,
                'author_neighb': 3,
                'author_cand_lim': 5,
                'track_neighb': 3,
                'track_cand_lim': 5
    }
    ```
    For example,
    
    <b>GET /recommend?id=<i>666</i>&base_l_neighb=<i>228</i>
    &author_cand_lim=<i>69</i></b>
    
    Note that such big parameter values may increase
execution time drastically!
4. <b>GET /check</b> - check the connection between RecSys and DB.