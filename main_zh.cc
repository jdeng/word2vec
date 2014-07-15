// OpenMP is required..
// g++-4.8 -ozh -fopenmp -std=c++0x -Ofast -march=native -funroll-loops  zh.cc  -lpthread

#include <iostream>
#include <initializer_list>
#include <string>
#include <set>

#include "word2vec.h"
#include "chars_zh.h"

int main(int argc, const char *argv[])
{
	Word2Vec<std::u16string> model(200);
	using Sentence = Word2Vec<std::u16string>::Sentence;
	using SentenceP = Word2Vec<std::u16string>::SentenceP;

	model.sample_ = 0;
    model.min_count_ = 3;
//	model.window_ = 10;
//	model.phrase_ = true;
	int n_workers = 4;

	::srand(::time(NULL));

	auto distance = [&model]() {
		while (1) {
			std::string s;
			std::cout << "\nFind nearest word for (:quit to break):";
			std::cin >> s;
			if (s == ":quit") break;
			auto p = model.most_similar(std::vector<std::u16string>{Cvt<std::u16string>::from_utf8(s)}, std::vector<std::u16string>(), 10);
			size_t i = 0;
			for (auto& v: p) {
				std::cout << i++ << " " << Cvt<std::u16string>::to_utf8(v.first) << " " << v.second << std::endl;
			}
		}
	};

	bool train = true, test = true;

	if (train) {
		std::vector<SentenceP> sentences;

		size_t count =0;
		const size_t max_sentence_len = 200;

		SentenceP sentence(new Sentence);
		std::ifstream in(argv[1]);
		while (true) {
			std::string s;
			in >> s;
			if (s.empty()) break;	
			
			std::u16string us = Cvt<std::u16string>::from_utf8(s);
			for (auto ch: us) {
				if (is_word(ch))
					sentence->tokens_.push_back(std::u16string(1, ch));
				if (! is_word(ch) || sentence->tokens_.size() == max_sentence_len) {
					if (ch == ' ' || sentence->tokens_.empty()) 
						continue;
					sentence->words_.reserve(sentence->tokens_.size());
					sentences.push_back(std::move(sentence));
					sentence.reset(new Sentence);
					continue;
				}
			}
		}
		
		if (!sentence->tokens_.empty())
			sentences.push_back(std::move(sentence));

		std::cout << sentences.size() << " sentences, " << std::accumulate(sentences.begin(), sentences.end(), (int)0, [](int x, const SentenceP& s) { return x + s->tokens_.size(); }) << " words loaded" << std::endl;

		auto cstart = std::chrono::high_resolution_clock::now();
		model.build_vocab(sentences);
		auto cend = std::chrono::high_resolution_clock::now();
		printf("load vocab: %.4f seconds model size: %d\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0, model.words_.size());

		cstart = cend;
		model.train(sentences, n_workers);
		cend = std::chrono::high_resolution_clock::now();
		printf("train: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

		cstart = cend;
		model.save("vectors.bin");
		model.save_text("vectors.txt");
		cend = std::chrono::high_resolution_clock::now();
		printf("save model: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);
	}

	if (test) {
		auto cstart = std::chrono::high_resolution_clock::now();
		model.load("vectors.bin");
		auto cend = std::chrono::high_resolution_clock::now();
		printf("load model: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

		distance();
	}

	return 0;
}

