import mysql.connector
from mysql.connector import errorcode
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import pickle
from flask import Flask, request, abort


class MashupRecSys:
    def __init__(self,username,password,host,port,db):
        self.cnx = None
        self.db_config = {
            'user': username,
            'password': password,
            'host': host,
            'port': port,
            'database': db
        }
        self.params_default = {
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
        self.scaler = StandardScaler()
        self.knn_model = pickle.load(open('knn_pretrained', 'rb'))
        with open('ind_to_mashup_id.npy', 'rb') as f:
            self.available_mashup_ids = np.load(f)

    def connect_to_db(self):
        try:
            self.cnx = mysql.connector.connect(**self.db_config)
            return True
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with the username or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)
        return False

    def query_db(self, query: str):
        if self.cnx and self.cnx.is_connected():
            with self.cnx.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
            return rows
        return None

    @staticmethod
    def multifeature_encoder(values, ids, num_unique_values=None, num_unique_ids=None):
        # inputs - vectors, not ndarrays
        values_enum = dict([(i[1], i[0]) for i in enumerate(np.unique(values))])
        if num_unique_values is None:
            num_unique_values = len(values_enum)
        if num_unique_ids is None:
            num_unique_ids = len(np.unique(ids))

        result = np.zeros((num_unique_ids, num_unique_values))
        prev_id, cur_ind = 0, -1
        for v, i in zip(values, ids):
            if i != prev_id:
                prev_id = i
                cur_ind += 1
            result[cur_ind][values_enum[v]] = 1

        return result

    def retrain_model(self, overwrite=True):
        def get_dataset():
            self.connect_to_db()
            self.available_mashup_ids = np.array(self.query_db(f'SELECT id FROM mashups'))
            id_lim = max(self.available_mashup_ids)[0]

            statuses = np.array(self.query_db(f'SELECT statuses FROM mashups WHERE id<{id_lim}'))
            durations = np.array(self.query_db(f'SELECT duration FROM mashups WHERE id<{id_lim}'))
            genres_raw = np.array(self.query_db(f'SELECT genre, mashup_id FROM mashups JOIN mashups_to_genres ON mashups.id=mashups_to_genres.mashup_id WHERE mashup_id<{id_lim}'))

            n_unique_values = int(self.query_db('SELECT COUNT(DISTINCT genre) FROM mashups_to_genres')[0][0])
            n_unique_ids = int(self.query_db(f'SELECT COUNT(id) FROM mashups WHERE id<{id_lim}')[0][0])
            genres = self.multifeature_encoder(genres_raw[:,0], genres_raw[:,1], n_unique_values, n_unique_ids)
            self.cnx.close()

            features = (statuses, durations, genres)
            for f in features:
                f[np.isnan(f)] = 0

            X = np.hstack(features)
            X_scaled = self.scaler.fit_transform(X)
            print(f'Successfully obtained a dataset with {np.shape(X)[0]} items and {np.shape(X)[1]} features.')
            return X_scaled

        self.knn_model = NearestNeighbors().fit(get_dataset())

        model_file = 'knn_pretrained'
        ind_map_file = 'ind_to_mashup_id.npy'
        if not overwrite:
            model_file = 'knn_retrained'
            ind_map_file = 'ind_to_mashup_id_new.npy'

        knn_pickle = open(model_file, 'wb')
        pickle.dump(self.knn_model, knn_pickle)
        knn_pickle.close()

        with open(ind_map_file, 'wb') as f:
            np.save(f, self.available_mashup_ids)

        print(f"Successfully saved the trained model to '{model_file}' and 'dataset ind -> db id' mapping to '{ind_map_file}'.")

    def get_rec_list(self, user_id, **kwargs):
        '''
        :param user_id: id of a user in the db
        :param kwargs: parameter values (if any one is not stated, it is set by default)
        :return: list of recommended mashups in form of db ids. number of elements:
        base_l_neighb + base_m_neighb + base_r_neighb + additional,
        where additional <=
        playlist_cand_lim * playlist_neighb+
        author_cand_lim * author_neighb+
        track_cand_lim * track_neighb
        '''

        def filter_already_liked(mashup_ids):
            likes = [i[0] for i in self.query_db(f'SELECT mashup_id FROM mashups_likes WHERE user_id={user_id}')]
            filtered_ids = []
            for i in mashup_ids:
                if i not in likes:
                    filtered_ids.append(i)
            return filtered_ids

        def get_content_data_point(mashup_id):
            status = np.array(self.query_db(f'SELECT statuses FROM mashups WHERE id={mashup_id}'))
            duration = np.array(self.query_db(f'SELECT duration FROM mashups WHERE id={mashup_id}'))
            genre_raw = np.array(self.query_db(
                f'SELECT genre FROM mashups JOIN mashups_to_genres ON mashups.id=mashups_to_genres.mashup_id WHERE mashup_id={mashup_id}'
            ))

            n_unique_values = int(self.query_db('SELECT COUNT(DISTINCT genre) FROM mashups_to_genres')[0][0])
            genre = self.multifeature_encoder(
                genre_raw.reshape(1, -1)[0], 
                np.array([mashup_id] * len(genre_raw)),
                n_unique_values
            )
            features = (status, duration, genre)
            for f in features:
                f[np.isnan(f)] = 0

            # each datapoint is of shape (1,num_of_features)
            x = np.hstack(features)
            return self.scaler.fit_transform(x)

        def fill_random_rem_mashups(cur_list, required_size):
            filter_already_liked(cur_list)
            cur_list = list(set(cur_list))
            while len(cur_list) < required_size:
                remainder = required_size - len(cur_list)
                cur_list += self.query_db(f'SELECT mashup_id FROM mashups ORDER BY RAND() LIMIT {remainder}')
                cur_list = list(set(cur_list))

        def get_pop_ids(liked_population_size, most_listened_population_size, recently_listened_population_size):
            '''
            Always returns a list with exactly l_pop_size+m_pop_size+r_pop_size distinct db mashup ids
            '''
            liked_population = self.query_db(
                f'SELECT mashup_id FROM mashups_likes WHERE user_id={user_id} ORDER BY RAND() LIMIT {liked_population_size}')
            most_listened_population = self.query_db(
                f"SELECT mashup_id FROM mashups_likes WHERE user_id={user_id} GROUP BY mashup_id ORDER BY COUNT(`time`) DESC, RAND() LIMIT {most_listened_population_size}")
            recently_listened_population = self.query_db(
                f"SELECT mashup_id FROM mashups_likes WHERE user_id={user_id} ORDER BY `time` DESC, RAND() LIMIT {recently_listened_population_size}")

            fill_random_rem_mashups(liked_population, liked_population_size)
            fill_random_rem_mashups(most_listened_population, most_listened_population_size)
            fill_random_rem_mashups(recently_listened_population, recently_listened_population_size)

            return liked_population, most_listened_population, recently_listened_population

        def get_pop_data(liked_population_ids, most_listened_population_ids, recently_listened_population_ids):
            liked_population = np.zeros((len(liked_population_ids), 16))
            most_listened_population = np.zeros((len(most_listened_population_ids), 16))
            recently_listened_population = np.zeros((len(recently_listened_population_ids), 16))

            # each datapoint is of shape (1,num_of_features)
            for i in range(len(liked_population_ids)):
                liked_population[i] = get_content_data_point(liked_population_ids[i][0])

            for i in range(len(most_listened_population_ids)):
                most_listened_population[i] = get_content_data_point(most_listened_population_ids[i][0])

            for i in range(len(recently_listened_population_ids)):
                recently_listened_population[i] = get_content_data_point(recently_listened_population_ids[i][0])

            return liked_population, most_listened_population, recently_listened_population

        def select_candidates_base(liked_population, most_listened_population, recently_listened_population,
                                   l_neighbors, m_neighbors, r_neighbors):
            # neighbor search (candidates are returned as indices stored in ids and corresponding to elements of X provided to the KNN)
            l_dist, l_candidates = self.knn_model.kneighbors(liked_population, l_neighbors)
            m_dist, m_candidates = self.knn_model.kneighbors(most_listened_population, m_neighbors)
            r_dist, r_candidates = self.knn_model.kneighbors(recently_listened_population, r_neighbors)

            # shapes of all the received arrays: (n_population, n_neighbors)
            return np.concatenate((l_candidates.flatten(), m_candidates.flatten(), r_candidates.flatten()), axis=None)

        def select_candidates_from_playlist(liked_population_ids, most_listened_population_ids,
                                            recently_listened_population_ids, n_neighbors, lim):
            # neighbor search based on playlist data (mashup-candidates are returned as indices stored in ids and corresponding to elements of X provided to the KNN)
            playlist_ids = set()
            # playlists including songs from the populations
            for pop in [liked_population_ids, most_listened_population_ids, recently_listened_population_ids]:
                for i in tqdm(pop, desc='Playlist ids from pop'):
                    for j in self.query_db(
                            f'SELECT playlist_id FROM playlists_to_mashups WHERE mashup_id = {i[0]} ORDER BY RAND() LIMIT {lim}'):
                        playlist_ids.add(j[0])

            # liked playlists
            for i in self.query_db(
                    f'SELECT playlist_id FROM playlists_likes WHERE user_id = {user_id} ORDER BY RAND() LIMIT {lim}'):
                playlist_ids.add(i[0])

            candidates = []
            for pi in tqdm(playlist_ids, desc='Playlist based recs'):
                mashup_ids = self.query_db(
                    f'SELECT mashup_id FROM playlists_to_mashups WHERE playlist_id = {pi} ORDER BY RAND() LIMIT {lim}')
                for mi in mashup_ids:
                    mashup_data = get_content_data_point(mi[0])
                    dist, neigh_inds = self.knn_model.kneighbors(mashup_data, n_neighbors)
                    for i in range(n_neighbors):
                        candidates.append((dist[0][i], neigh_inds[0][i]))

            return set([pair[1] for pair in sorted(candidates, key=lambda pair: pair[0])][:lim])

        def select_candidates_from_author(liked_population_ids, most_listened_population_ids,
                                          recently_listened_population_ids, n_neighbors, lim):
            # neighbor search based on mashup author data (mashup-candidates are returned as indices stored in ids and corresponding to elements of X provided to the KNN)
            author_ids = set()
            # authors of mashups from the populations
            for pop in [liked_population_ids, most_listened_population_ids, recently_listened_population_ids]:
                for i in tqdm(pop, desc='Author ids from pop'):
                    for j in self.query_db(
                            f'SELECT user_id FROM mashups_to_authors WHERE mashup_id={i[0]} ORDER BY RAND() LIMIT {lim}'):
                        author_ids.add(j[0])

            # liked mashups - already in liked_population

            candidates = []
            for ai in tqdm(author_ids, desc='Author based recs'):
                mashup_ids = self.query_db(
                    f'SELECT mashup_id FROM mashups_to_authors WHERE user_id={ai} ORDER BY RAND() LIMIT {lim}')
                for mi in mashup_ids:
                    mashup_data = get_content_data_point(mi[0])
                    dist, neigh_inds = self.knn_model.kneighbors(mashup_data, n_neighbors)
                    for i in range(n_neighbors):
                        candidates.append((dist[0][i], neigh_inds[0][i]))

            return set([pair[1] for pair in sorted(candidates, key=lambda pair: pair[0])][:lim])

        def select_candidates_from_track(liked_population_ids, most_listened_population_ids,
                                         recently_listened_population_ids, n_neighbors, lim):
            # neighbor search based on data about tracks that the mashup consists of (mashup-candidates are returned as indices stored in ids and corresponding to elements of X provided to the KNN)
            track_ids = set()
            # get tracks from mashups from the populations (liked mashups - already in liked_population)
            for pop in [liked_population_ids, most_listened_population_ids, recently_listened_population_ids]:
                for i in tqdm(pop, desc='Track ids from pop'):
                    for j in self.query_db(
                            f'SELECT track_id FROM mashups_to_tracks WHERE mashup_id={i[0]} ORDER BY RAND() LIMIT {lim}'):
                        track_ids.add(j[0])

            # get the tracks' authors
            author_ids = set()
            for i in track_ids:
                for j in self.query_db(
                        f'SELECT author_id FROM tracks_to_authors WHERE track_id={i} ORDER BY RAND() LIMIT {lim}'):
                    author_ids.add(j[0])

            # get other tracks of the authors
            for i in author_ids:
                for j in self.query_db(
                        f'SELECT track_id FROM tracks_to_authors WHERE author_id={i} ORDER BY RAND() LIMIT {lim}'):
                    track_ids.add(j[0])

            # recommend other mashups that include these tracks
            candidates = []
            for ti in tqdm(track_ids, desc='Track based recs'):
                mashup_ids = self.query_db(
                    f'SELECT mashup_id FROM mashups_to_tracks WHERE track_id={ti} ORDER BY RAND() LIMIT {lim}')
                for mi in mashup_ids:
                    mashup_data = get_content_data_point(mi[0])
                    dist, neigh_inds = self.knn_model.kneighbors(mashup_data, n_neighbors)
                    for i in range(n_neighbors):
                        candidates.append((dist[0][i], neigh_inds[0][i]))
                        
            return set([pair[1] for pair in sorted(candidates, key=lambda pair: pair[0])][:lim])

        params = self.params_default
        for key, value in kwargs.items():
            params[key] = value

        self.connect_to_db()

        population_ids = get_pop_ids(params['liked_pop_size'], params['most_listened_pop_size'], params['recently_listened_pop_size'])
        populations = get_pop_data(*population_ids)

        # get base (already filtered from already liked)
        base_cand = select_candidates_base(*populations, params['base_l_neighb'], params['base_m_neighb'], params['base_r_neighb'])

        # get additional without already liked
        playlist_cand = filter_already_liked(select_candidates_from_playlist(*population_ids, params['playlist_neighb'], params['playlist_cand_lim']))
        author_cand = filter_already_liked(select_candidates_from_author(*population_ids, params['author_neighb'], params['author_cand_lim']))
        track_cand = filter_already_liked(select_candidates_from_track(*population_ids, params['track_neighb'], params['track_cand_lim']))
        
        self.cnx.close()

        total_cand = list(base_cand) + playlist_cand + author_cand + track_cand
        res = []
        for ind in total_cand:
            if ind >= len(self.available_mashup_ids):
                print(f'Warning! Recommended mashup index {ind} is out of bounds.')
            else:
                res.append(int(self.available_mashup_ids[ind][0]))

        print(f'Successfully generated a recommendation list for user {user_id}.')
        return res


app = Flask(__name__)
app.config.from_object(__name__)


@app.route('/')
def init_recsys():
    try:
        username = request.args.get('username')
        password = request.args.get('password')
        host = request.args.get('host')
        port = int(request.args.get('port'))
        db = request.args.get('db')
        app.recsys = MashupRecSys(username,password,host,port,db)
        return '<h1>200 Successfully initialized the RecSys object.</h1>', 200
    except:
        abort(401)


@app.route('/train', methods=['GET','POST'])
def train():
    try:
        app.recsys.retrain_model()
        return '<h1>200 Successfully started training.</h1>', 200
    except:
        abort(500)


@app.route('/recommend', methods=['GET','POST'])
def recommend():
    user_id = 0
    params = dict()
    try:
        user_id = int(request.args.get('id'))
        for key in app.recsys.params_default.keys():
            if request.args.get(key) is not None:
                params[key] = int(request.args.get(key))
    except:
        abort(400)

    try:
        lst = app.recsys.get_rec_list(user_id, **params)
        return {"recs": lst}, 200
    except:
        abort(500)


if __name__ == '__main__':
    # recsys = MashupRecSys(username='root', password='x155564py', host='127.0.0.1', port=3307, db='smashup')
    # recsys.retrain_model()
    # res = recsys.get_rec_list(2)
    app.run(port=5000)