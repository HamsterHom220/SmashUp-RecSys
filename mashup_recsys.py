import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import pickle
from flask import Flask, request, abort
import os
from dotenv import load_dotenv
from scipy.spatial import distance


class MashupRecSys:
    def __init__(self, local=False):
        self.local = local
        self.cnx = None
        self.params_default = {
            'liked_pop_size': 3,
            'most_listened_pop_size': 5,
            'recently_listened_pop_size': 10,
            'base_l_neighb': 10,
            'base_m_neighb': 5,
            'base_r_neighb': 5,
            'playlist_neighb': 5,
            'playlist_cand_lim': 3,
            'author_neighb': 5,
            'author_cand_lim': 3,
            'track_neighb': 5,
            'track_cand_lim': 3
        }
        self.scaler = StandardScaler()
        self.knn_model = pickle.load(open('knn_pretrained', 'rb'))
        with open('ind_to_mashup_id.npy', 'rb') as f:
            self.available_mashup_ids = np.load(f)

    def connect_to_db(self):
        if self.local:
            dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
            load_dotenv(dotenv_path)

        self.cnx = mysql.connector.connect(
            user=os.environ.get('MYSQL_USER'), password=os.environ.get('MYSQL_PASSWORD'),
            host=os.environ.get('MYSQL_HOST'), port=int(os.environ.get('MYSQL_PORT')),
            database=os.environ.get('MYSQL_DATABASE')
        )

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

    def get_dataset(self):
        self.connect_to_db()
        self.available_mashup_ids = np.array(self.query_db(f'SELECT id FROM mashups'))
        id_lim = max(self.available_mashup_ids)[0]

        statuses = np.array(self.query_db(f'SELECT statuses FROM mashups WHERE id<{id_lim}'))
        durations = np.array(self.query_db(f'SELECT duration FROM mashups WHERE id<{id_lim}'))
        genres_raw = np.array(self.query_db(
            f'SELECT genre, mashup_id FROM mashups JOIN mashups_to_genres ON mashups.id=mashups_to_genres.mashup_id WHERE mashup_id<{id_lim}'))

        n_unique_values = int(self.query_db('SELECT COUNT(DISTINCT genre) FROM mashups_to_genres')[0][0])
        n_unique_ids = int(self.query_db(f'SELECT COUNT(id) FROM mashups WHERE id<{id_lim}')[0][0])
        genres = self.multifeature_encoder(genres_raw[:, 0], genres_raw[:, 1], n_unique_values, n_unique_ids)
        self.cnx.close()

        features = (statuses, durations, genres)
        for f in features:
            f[np.isnan(f)] = 0

        X = np.hstack(features)
        X_scaled = self.scaler.fit_transform(X)
        print(f'Successfully obtained a dataset with {np.shape(X)[0]} items and {np.shape(X)[1]} features.')
        return X_scaled

    def retrain_model(self, overwrite=True):
        self.knn_model = NearestNeighbors(metric='cosine').fit(self.get_dataset())

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

    def get_content_data_point(self, mashup_id):
        # print('GET CONTENT DATA POINT')
        # print('mashup id:', mashup_id)
        status = np.array(self.query_db(f'SELECT statuses FROM mashups WHERE id={mashup_id}'))
        duration = np.array(self.query_db(f'SELECT duration FROM mashups WHERE id={mashup_id}'))
        genre_raw = np.array(self.query_db(
            f'SELECT genre FROM mashups JOIN mashups_to_genres ON mashups.id=mashups_to_genres.mashup_id WHERE mashup_id={mashup_id}'
        ))

        n_unique_values = int(self.query_db('SELECT COUNT(DISTINCT genre) FROM mashups_to_genres')[0][0])
        genre = self.multifeature_encoder(
            genre_raw.reshape(1, -1)[0],
            np.array([mashup_id] * len(genre_raw)),
            n_unique_values,
            1
        )
        features = (status, duration, genre)
        for f in features:
            f[np.isnan(f)] = 0

        # each datapoint is of shape (1,num_of_features)
        x = np.hstack(features)
        # print('datapoint:', self.scaler.transform(x))
        # print('========================')
        return self.scaler.transform(x)

    def get_rec_list(self, user_id, rec_lim, **kwargs):
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


        def fill_random_rem_mashups(cur_list, required_size, numbers=False):
            filter_already_liked(cur_list)
            if not numbers:
                cur_list = [i[0] for i in set(cur_list)]
            else:
                cur_list = list(set(cur_list))
            while len(cur_list) < required_size:
                remainder = required_size - len(cur_list)
                [cur_list.append(i[0]) for i in self.query_db(f'SELECT id FROM mashups ORDER BY RAND() LIMIT {remainder}')]
                cur_list = list(set(cur_list))
            return cur_list

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

            # most_listened_population = list(set(most_listened_population)-set(liked_population))
            recently_listened_population = list(set(recently_listened_population)-set(most_listened_population)) #-set(liked_population)

            liked_population = fill_random_rem_mashups(liked_population, liked_population_size)
            most_listened_population = fill_random_rem_mashups(most_listened_population, 5)
            recently_listened_population = fill_random_rem_mashups(recently_listened_population, 5)

            print('Population:')
            self.print_mashup_data(liked_population)
            self.print_mashup_data(most_listened_population)
            self.print_mashup_data(recently_listened_population)

            return liked_population, most_listened_population, recently_listened_population

        def get_pop_data(liked_population_ids, most_listened_population_ids, recently_listened_population_ids):
            liked_population = np.zeros((len(liked_population_ids), 16))
            most_listened_population = np.zeros((len(most_listened_population_ids), 16))
            recently_listened_population = np.zeros((len(recently_listened_population_ids), 16))

            # each datapoint is of shape (1,num_of_features)
            for i in range(len(liked_population_ids)):
                liked_population[i] = self.get_content_data_point(liked_population_ids[i])

            for i in range(len(most_listened_population_ids)):
                most_listened_population[i] = self.get_content_data_point(most_listened_population_ids[i])

            for i in range(len(recently_listened_population_ids)):
                recently_listened_population[i] = self.get_content_data_point(recently_listened_population_ids[i])

            return liked_population, most_listened_population, recently_listened_population

        def select_candidates_base(liked_population, most_listened_population, recently_listened_population,
                                   l_neighbors, m_neighbors, r_neighbors):
            # neighbor search (candidates are returned as indices stored in ids and corresponding to elements of X provided to the KNN)
            # print('Liked population: ', liked_population)
            # print('Most listened population: ', most_listened_population)
            # print('Recently listened population: ', recently_listened_population)

            l_dist, l_candidates = self.knn_model.kneighbors(liked_population, l_neighbors)
            m_dist, m_candidates = self.knn_model.kneighbors(most_listened_population, m_neighbors)
            r_dist, r_candidates = self.knn_model.kneighbors(recently_listened_population, r_neighbors)

            l_candidates = set(l_candidates.flatten())
            m_candidates = set(m_candidates.flatten())
            r_candidates = set(r_candidates.flatten())

            m_candidates = m_candidates - l_candidates
            r_candidates = r_candidates - m_candidates - l_candidates

            l_candidates = list(l_candidates)
            m_candidates = list(m_candidates)
            r_candidates = list(r_candidates)

            m_candidates = fill_random_rem_mashups(m_candidates, m_neighbors, True)
            r_candidates = fill_random_rem_mashups(r_candidates, r_neighbors, True)

            print('Neighbors of liked:', l_candidates)
            print('Neighbors of most listened:', m_candidates)
            print('Neighbors of recently listened:', r_candidates)

            # shapes of all the received arrays: (n_population, n_neighbors)
            return l_candidates+m_candidates+r_candidates

        def select_candidates_from_playlist(liked_population_ids, most_listened_population_ids,
                                            recently_listened_population_ids, n_neighbors, lim):
            # neighbor search based on playlist data (mashup-candidates are returned as indices stored in ids and corresponding to elements of X provided to the KNN)
            playlist_ids = set()
            # playlists including songs from the populations
            for pop in [liked_population_ids, most_listened_population_ids, recently_listened_population_ids]:
                for i in tqdm(pop, desc='Playlist ids from pop'):
                    for j in self.query_db(
                            f'SELECT playlist_id FROM playlists_to_mashups WHERE mashup_id = {i} ORDER BY RAND() LIMIT {lim}'):
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
                    mashup_data = self.get_content_data_point(mi[0])
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
                            f'SELECT user_id FROM mashups_to_authors WHERE mashup_id={i} ORDER BY RAND() LIMIT {lim}'):
                        author_ids.add(j[0])

            # liked mashups - already in liked_population

            candidates = []
            for ai in tqdm(author_ids, desc='Author based recs'):
                mashup_ids = self.query_db(
                    f'SELECT mashup_id FROM mashups_to_authors WHERE user_id={ai} ORDER BY RAND() LIMIT {lim}')
                for mi in mashup_ids:
                    mashup_data = self.get_content_data_point(mi[0])
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
                            f'SELECT track_id FROM mashups_to_tracks WHERE mashup_id={i} ORDER BY RAND() LIMIT {lim}'):
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
                    mashup_data = self.get_content_data_point(mi[0])
                    dist, neigh_inds = self.knn_model.kneighbors(mashup_data, n_neighbors)
                    for i in range(n_neighbors):
                        candidates.append((dist[0][i], neigh_inds[0][i]))
                        
            return set([pair[1] for pair in sorted(candidates, key=lambda pair: pair[0])][:lim])

        params = self.params_default
        for key, value in kwargs.items():
            params[key] = value

        self.get_dataset()
        self.connect_to_db()

        population_ids = get_pop_ids(params['liked_pop_size'], params['most_listened_pop_size'], params['recently_listened_pop_size'])
        populations = get_pop_data(*population_ids)

        # get base (already filtered from already liked)
        base_cand = select_candidates_base(*populations, params['base_l_neighb'], params['base_m_neighb'], params['base_r_neighb'])

        # get additional without already liked
        playlist_cand = filter_already_liked(select_candidates_from_playlist(*population_ids, params['playlist_neighb'], params['playlist_cand_lim']))
        author_cand = filter_already_liked(select_candidates_from_author(*population_ids, params['author_neighb'], params['author_cand_lim']))
        track_cand = filter_already_liked(select_candidates_from_track(*population_ids, params['track_neighb'], params['track_cand_lim']))

        playlist_cand = set(playlist_cand)-set(base_cand)
        author_cand = set(author_cand)-set(base_cand)-set(playlist_cand)
        track_cand = set(track_cand)-set(playlist_cand)-set(author_cand)-set(base_cand)

        playlist_cand = fill_random_rem_mashups(list(playlist_cand), params['playlist_neighb']*params['playlist_cand_lim'])
        author_cand = fill_random_rem_mashups(list(author_cand), params['author_neighb']*params['author_cand_lim'])
        track_cand = fill_random_rem_mashups(list(track_cand), params['track_neighb']*params['track_cand_lim'])

        print('Base cands:',base_cand)
        print('Playlist cands:', playlist_cand)
        print('Author cands:', author_cand)
        print('Track cands:', track_cand)

        total_cand = list(base_cand) + list(playlist_cand) + list(author_cand) + list(track_cand)
        res = []
        for ind in total_cand:
            if ind >= len(self.available_mashup_ids):
                print(f'Warning! Recommended mashup index {ind} is out of bounds.')
            else:
                res.append(int(self.available_mashup_ids[ind][0]))

        print(f'Successfully generated a recommendation list for user {user_id}.')
        sorted_res = self.sort_mashup_list(list(set(res)), np.concatenate([*populations],axis=0))
        self.cnx.close()
        return sorted_res[:-(len(sorted_res)-rec_lim)]

    def print_mashup_data(self, mashup_ids, was_connected=True):
        if not was_connected:
            self.connect_to_db()
        data = []
        for i in mashup_ids:
            data.append(self.query_db(f'SELECT id, name, genre FROM mashups JOIN mashups_to_genres ON mashups.id = mashup_id WHERE mashup_id={i}'))
        if not was_connected:
            self.cnx.close()
        [print(row[0]) for row in data]

    def sort_mashup_list(self, mashup_ids, population_data):
        center = np.mean(population_data)
        dists = np.zeros(len(mashup_ids))
        for i in range(len(mashup_ids)):
            dists[i] = distance.cosine(center.flatten(), self.get_content_data_point(mashup_ids[i]).flatten())
        res = []
        for p in sorted(zip(mashup_ids,dists), key=lambda p: p[1]):
            res.append(p[0])
        return res


app = Flask(__name__)
app.config.from_object(__name__)


@app.route('/')
def init_recsys():
    try:
        app.recsys = MashupRecSys()
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
    rec_lim = 15
    params = dict()
    try:
        user_id = int(request.args.get('id'))
        rec_lim = int(request.args.get('lim'))
        for key in app.recsys.params_default.keys():
            if request.args.get(key) is not None:
                params[key] = int(request.args.get(key))
    except:
        abort(400)

    try:
        lst = app.recsys.get_rec_list(user_id, rec_lim, **params)
        return {"recs": lst}, 200
    except:
        abort(500)


@app.route('/check', methods=['GET'])
def check():
    app.recsys.connect_to_db()
    app.recsys.cnx.close()
    return '<h1>200 DB connection is OK.</h1>', 200


if __name__ == '__main__':
    # recsys = MashupRecSys(True)
    # res = recsys.get_rec_list(user_id=123, rec_lim=20)
    # recsys.print_mashup_data(res, False)
    app.run(host='0.0.0.0', port=5000, debug=True) # values reserved for container
