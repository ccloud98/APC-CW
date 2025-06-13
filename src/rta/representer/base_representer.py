import torch
import torch.nn as nn
import numpy as np
from src.rta.utils import get_device

class BaseEmbeddingRepresenter(nn.Module):
    '''
      Base class for representers. Can store all relevant embeddings for a given song
      (song embedding, artist embedding, album embedding, etc...)   
      
    '''
    def __init__(self,
                 data_manager,
                 emb_dim):
        super(BaseEmbeddingRepresenter, self).__init__()
        self.dev = get_device()
        n_tokens = data_manager.n_tracks + 2  # 1 for PAD, 1 for MASK
        self.data_manager = data_manager
        
        # 디버깅을 위한 정보 출력
        print(f"[INFO] n_tokens: {n_tokens}, song_embeddings 크기: {np.load(data_manager.song_embeddings_path).shape}")
        print(f"[INFO] song_pop 길이: {len(data_manager.song_pop)}, song_duration 길이: {len(data_manager.song_duration)}")
        
        # 노래 임베딩
        enhanced_embedding = np.zeros((n_tokens, emb_dim))
        song_emb = np.load(data_manager.song_embeddings_path)
        song_length = min(song_emb.shape[0], n_tokens-2)
        enhanced_embedding[1:1+song_length, :] = song_emb[:song_length, :emb_dim]
        print(f"[INFO] 노래 임베딩 로드: {song_emb.shape}")
        print(f"[INFO] 노래 임베딩 초기화 완료: {song_length}/{n_tokens-2} 아이템")
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(enhanced_embedding), freeze=False).float()

        # 앨범 임베딩
        n_albums = len(data_manager.album_ids) + 2  # 1 for PAD, 1 for MASK
        self.song_album = np.zeros(n_tokens)
        album_length = min(len(data_manager.song_album), n_tokens-2)
        self.song_album[1:1+album_length] = data_manager.song_album[:album_length] + 1  # each album is offset because of PAD
        self.song_album[-1] = n_albums + 1
        self.song_album = torch.LongTensor(self.song_album).to(self.dev)
        self.song_album.require_grad = False
        
        album_enhanced_embedding = np.zeros((n_albums, emb_dim))
        album_emb = np.load(data_manager.album_embeddings_path)
        album_emb_length = min(album_emb.shape[0], n_albums-2)
        album_enhanced_embedding[1:1+album_emb_length, :] = album_emb[:album_emb_length, :emb_dim]
        self.album_embedding = nn.Embedding.from_pretrained(torch.tensor(album_enhanced_embedding), freeze=False).float()

        # 아티스트 임베딩
        n_artists = len(data_manager.artist_ids) + 2  # 1 for PAD, 1 for MASK
        self.song_artist = np.zeros(n_tokens)
        artist_length = min(len(data_manager.song_artist), n_tokens-2)
        self.song_artist[1:1+artist_length] = data_manager.song_artist[:artist_length] + 1  # each artist is offset because of PAD
        self.song_artist[-1] = n_artists + 1
        self.song_artist = torch.LongTensor(self.song_artist).to(self.dev)
        self.song_artist.require_grad = False
        
        artist_enhanced_embedding = np.zeros((n_artists, emb_dim))
        artist_emb = np.load(data_manager.artist_embeddings_path)
        artist_emb_length = min(artist_emb.shape[0], n_artists-2)
        artist_enhanced_embedding[1:1+artist_emb_length, :] = artist_emb[:artist_emb_length, :emb_dim]
        self.artist_embedding = nn.Embedding.from_pretrained(torch.tensor(artist_enhanced_embedding), freeze=False).float()

        # 인기도 임베딩
        self.song_pop = np.zeros(n_tokens)
        pop_length = min(len(data_manager.song_pop), n_tokens-2)
        self.song_pop[1:1+pop_length] = data_manager.song_pop[:pop_length]  # each pop is offset because of PAD
        self.song_pop = torch.LongTensor(self.song_pop).to(self.dev)
        self.song_pop.require_grad = False
        
        pop_emb = np.load(data_manager.pop_embeddings_path)
        self.pop_embedding = nn.Embedding.from_pretrained(torch.tensor(pop_emb[:, :emb_dim]), freeze=False).float()

        # 재생 시간 임베딩
        self.song_dur = np.zeros(n_tokens)
        dur_length = min(len(data_manager.song_duration), n_tokens-2)
        self.song_dur[1:1+dur_length] = data_manager.song_duration[:dur_length]  # each dur is offset because of PAD
        self.song_dur = torch.LongTensor(self.song_dur).to(self.dev)
        self.song_dur.require_grad = False
        
        dur_emb = np.load(data_manager.dur_embeddings_path)
        self.dur_embedding = nn.Embedding.from_pretrained(torch.tensor(dur_emb[:, :emb_dim]), freeze=False).float()

    def forward(self, x):
        return self.embedding(x)

    def compute_all_representations(self):
        # Computes representations for all songs in the dataset. This is done prior to making recommendations.
        with torch.no_grad():
            step = 100000
            self.eval()
            current_index = 0
            max_index = self.data_manager.n_tracks + 1
            fusion_embeddings = torch.zeros(self.embedding.weight.shape).to(self.dev)
            while current_index < max_index:
                next_index = min(current_index + step, max_index)
                input_test = torch.LongTensor(range(current_index, next_index)).to(self.dev).unsqueeze(1)
                X_rep = self.forward(input_test.to(self.dev))
                fusion_embeddings[current_index: next_index] = X_rep.detach().squeeze()
                current_index = next_index
        return fusion_embeddings