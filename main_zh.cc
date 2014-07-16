// OpenMP is required..
// g++-4.8 -ozh -fopenmp -std=c++0x -Ofast -march=native -funroll-loops  main_zh.cc  -lpthread

#include <iostream>
#include <initializer_list>
#include <string>
#include <set>
#include <vector>

// #include <Magick++.h>
// #include "spectrum.inl"

#include "word2vec.h"
#include "rbm.h"

using Model = Word2Vec<std::u16string>;
using Sentence = Model::Sentence;
using SentenceP = Model::SentenceP;

void standardize(std::vector<Vector>& vs) {
	if (vs.size() <= 1) return;

	Vector m(vs[0].size()), d(vs[0].size());
	for (auto& x: vs) v::add(m, x);	
	v::scale(m, 1.0 / vs.size());

	for (auto& x: vs) {
		v::saxpy(x, -1.0, m);
		v::sax2(d, x);
	}

	v::scale(d, 1.0 / (vs.size() - 1));
	for (auto& i: d) i = 1.0 / sqrt(i); //sqrt(d);

	for (auto& x: vs) v::multiply(x, d);
}

const std::u16string MARKER = u"#m#";
std::vector<SentenceP> load_sentences(const std::string& path, bool with_marker, bool with_tag) {
	auto is_word = [](char16_t ch) { return ch >= 0x4e00 && ch <= 0x9fff; };
	auto close_tag = [](SentenceP& sentence) {
		Model::Tag& t = sentence->tags_.back();
		if (t == Model::B) t = Model::S;
		else if (t == Model::M) t = Model::E;
	};

		size_t count =0;
		const size_t max_sentence_len = 200;
		std::vector<SentenceP> sentences;

		SentenceP sentence(new Sentence);
		std::ifstream in(path);
		while (true) {
			std::string s;
			in >> s;
			if (s.empty()) break;	
			
			std::u16string us = Cvt<std::u16string>::from_utf8(s);
			for (auto ch: us) {
				if (is_word(ch)) {
					if (sentence->tokens_.empty() && with_marker) 
						sentence->tokens_.push_back(MARKER);
					sentence->tokens_.push_back(std::u16string(1, ch));

					if (with_tag) {
						if (sentence->tags_.empty())
							sentence->tags_.push_back(Model::B);
						else {
							auto& t = sentence->tags_.back();
							Model::Tag nt = (t == Model::S|| t == Model::E)? Model::B: Model::M;
							sentence->tags_.push_back(nt);
						}
					}
				}
				if (! is_word(ch) || sentence->tokens_.size() == max_sentence_len) {
					if (sentence->tokens_.empty()) continue;
					if (with_tag) close_tag(sentence);

					if (ch == u'，' || ch == u'、') continue;
					if (with_marker) sentence->tokens_.push_back(MARKER);
					sentence->words_.reserve(sentence->tokens_.size());
					sentences.push_back(std::move(sentence));
					sentence.reset(new Sentence);
				}
			}

			if (!sentence->tokens_.empty() && with_tag) close_tag(sentence);
		}
		
		if (!sentence->tokens_.empty()) {
			if (with_tag) close_tag(sentence);
			if (with_marker) sentence->tokens_.push_back(MARKER);
			sentences.push_back(std::move(sentence));
		}

	return sentences;
}

std::vector<Vector> generate_samples(const Model& model, const SentenceP& sentence, int window = 5) {
	const std::vector<float>& marker = model.word_vector(MARKER);
	if (marker.empty()) return std::vector<Vector>{};

	size_t n_tokens = sentence->tokens_.size();
	size_t vecsize = model.word_vector_size();
	Vector tmp((n_tokens + window) * vecsize);
	for (int i=0; i<window/2; ++i) 
		std::copy(marker.begin(), marker.end(), tmp.data() + i * vecsize);
	for (size_t i=0; i<n_tokens; ++i) {
		auto& s = sentence->tokens_[i];
		auto& w = model.word_vector(s);
		auto& cur = (w.empty()? marker: w);
		std::copy(cur.begin(), cur.end(), tmp.data() + (i + window/2) * vecsize);
	}
	for (int i=0; i<window/2; ++i) 
		std::copy(marker.begin(), marker.end(), tmp.data() + (i + window/2 + n_tokens) * vecsize);

	std::vector<Vector> samples;
	samples.reserve(n_tokens);
	for (size_t i=0; i<n_tokens; ++i)
		samples.emplace_back(tmp.data() + i * vecsize, tmp.data() + (i + window) * vecsize);

	return samples;
}


int main(int argc, const char *argv[])
{
	Model model(100);
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
		std::vector<SentenceP> sentences = load_sentences(argv[1], true, false);

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

#if 0
    // initialize pallet
    Magick::InitializeMagick(*argv);
    std::vector<Magick::Color> pallet;
    for (auto& rgb: _pallet) pallet.push_back(Magick::Color(rgb.r, rgb.g, rgb.b));
#endif

	bool build_net = true;
	if (build_net) {
		int window = 5;
		model.load("vectors.bin");

		std::vector<SentenceP> sentences = load_sentences(argv[1], false, true);
		std::vector<Vector> inputs, targets;
		for (auto& sentence: sentences) {
			auto samples = generate_samples(model, sentence, window);
			std::move(samples.begin(), samples.end(), std::back_inserter(inputs));
			auto& tags = sentence->tags_;
			for (auto t: tags) {
				std::vector<float> tv(4);
				tv[t] = 1.0;
				targets.emplace_back(std::move(tv));
			}
		}
		standardize(inputs);

		std::cout << inputs.size() << " inputs, " << targets.size() << " targets." << std::endl;
		for (size_t i=0; i<100000; i+= 1000) {
			std::cout << i << ":";
			for (auto& x: inputs[i]) std::cout << x << ", "; std::cout << std::endl;
			for (auto& x: targets[i]) std::cout << x << ", "; std::cout << std::endl;
			std::cout << std::endl;
		}

		auto progress = [&](DeepBeliefNet& dbn) {
			static int i = 0;
			std::string name = "dbn-" + std::to_string(i++);
			std::ofstream f(name + ".dat", std::ofstream::binary);

#if 0
		int width = 0, height = 0;
		Vector pixels;
		dbn.to_image(pixels, width, height);

		Magick::Image img(Magick::Geometry(width * 2, height * 2), Magick::Color(255,255,255));
		for (size_t x=0; x < width * 2; ++x) {
				for (size_t y=0; y < height * 2; ++y) {
					int i =  int(abs(pixels[int(y / 2 * width + x / 2)] * 255));
					if (i > 255 || i < 0) i = 255;
					img.pixelColor(x, y, pallet[i]);
				}
		}
		std::string fn = name + ".png";
		img.write(fn.c_str());
#endif

			dbn.store(f);
		};
	
		DeepBeliefNet dbn;

		dbn.build(std::vector<int>{(int)inputs[0].size(), 500, 300, 4});
//		dbn.rbms_[0]->continuous_ = true;
		auto& rbm = dbn.output_layer();
		rbm->type_ = RBM::Type::EXP;

	  	std::default_random_engine eng(::time(NULL));
  		std::normal_distribution<double> rng(0.0, 1.0);

		LRBM::Conf conf;

		bool resume = false;
		if (resume) {
			std::ifstream f("dbn.dat", std::ifstream::binary);
			dbn.load(f);
			conf.max_epoch_ = 2; conf.max_batches_ = 300; conf.batch_size_ = 200;
		}
		else {
			conf.max_epoch_ = 5; conf.max_batches_ = 300; conf.batch_size_ = 200;
			dbn.pretrain(inputs, conf, progress);
		}

		conf.max_epoch_ = 10; conf.max_batches_ /= 5; conf.batch_size_ *= 5;
		dbn.backprop(inputs, targets, conf, progress);

		std::ofstream f("dbn.dat", std::ofstream::binary);
		dbn.store(f);
	}

	bool test_net = false;
	if (test_net) {
	}

	return 0;
}

