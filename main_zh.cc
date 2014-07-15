// OpenMP is required..
// g++-4.8 -ozh -fopenmp -std=c++0x -Ofast -march=native -funroll-loops  main_zh.cc  -lpthread

#include <iostream>
#include <initializer_list>
#include <string>
#include <set>

#include "word2vec.h"

int main(int argc, const char *argv[])
{
	using Model = Word2Vec<std::u16string>;
	using Sentence = Model::Sentence;
	using SentenceP = Model::SentenceP;

	Model model(200);
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

	auto is_word = [](char16_t ch) { return ch >= 0x4e00 && ch <= 0x9fff; };
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
				if (is_word(ch)) {
					sentence->tokens_.push_back(std::u16string(1, ch));
					if (sentence->tags_.empty())
						sentence->tags_.push_back(Model::B);
					else {
						auto& t = sentence->tags_.back();
						Model::Tag nt = (t == Model::S|| t == Model::E)? Model::B: Model::M;
						sentence->tags_.push_back(nt);
					}
				}
				if (! is_word(ch) || sentence->tokens_.size() == max_sentence_len) {
					if (sentence->tokens_.empty()) continue;

					Model::Tag& t = sentence->tags_.back();
					if (t == Model::B) t = Model::S;
					else if (t == Model::M) t = Model::E;

					if (ch == u'，') continue;
					sentence->words_.reserve(sentence->tokens_.size());
					sentences.push_back(std::move(sentence));
					sentence.reset(new Sentence);
				}
			}

			if (!sentence->tokens_.empty()) {
				Model::Tag& t = sentence->tags_.back();
				if (t == Model::B) t = Model::S;
				else if (t == Model::M) t = Model::E;
			}
		}
		
		if (!sentence->tokens_.empty())
			sentences.push_back(std::move(sentence));
#if 0
		for (size_t i=0; i<sentences.size(); i += 1000) {
			auto s = sentences[i];
			for (auto w: s->tokens_) std::cout << Cvt<std::u16string>::to_utf8(w); std::cout << std::endl;
			for (auto t: s->tags_) std::cout << Model::tag_string(t); std::cout << std::endl;
		}
#endif

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

