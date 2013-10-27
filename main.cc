// Mac OS X:
// clang++ -ow2v -march=native -Ofast -funroll-loops -std=c++11 -stdlib=libc++ main.cc
// g++-4.8 -ow2v -Wa,-q -std=c++0x -Ofast -march=native -funroll-loops  main.cc 
// Linux:
// g++ -ow2v -std=c++0x -Ofast -march=native -funroll-loops  main.cc  -lpthread

#include "word2vec.h"
#include <iostream>

int accuracy(Model& model, std::string questions, int restrict_vocab = 30000) {
	std::ifstream in(questions);
	std::string line;
	auto lower = [](std::string& data) { std::transform(data.begin(), data.end(), data.begin(), ::tolower);};
	size_t count = 0, correct = 0, ignore = 0, almost_correct = 0;
	const int topn = 10;
	while (std::getline(in, line)) {
		if (line[0] == ':') {
			printf("%s\n", line.c_str());
			continue;
		}

		std::istringstream iss(line);
		std::string a, b, c, expected;
		iss >> a >> b >> c >> expected;
		lower(a); lower(b); lower(c); lower(expected);
		
		if (!model.has(a) || !model.has(b) || !model.has(c) || !model.has(expected)) {
			printf("unhandled: %s %s %s %s\n", a.c_str(), b.c_str(), c.c_str(), expected.c_str());
			++ignore;
			continue;
		}

		++count;
		std::vector<std::string> positive{b, c}, negative{a};
		auto predict = model.most_similar(positive, negative, topn);
		if (predict[0].first == expected) { ++ correct; ++almost_correct; }
		else {
			bool found = false;
			for (auto& v: predict) { 
				if (v.first == expected) { found = true; break; }
			}
			if (found) ++almost_correct;
			else printf("predicted: %s, expected: %s\n", predict[0].first.c_str(), expected.c_str());
		}
	}

	if (count > 0) printf("predict %lu out of %lu (%f%%), almost correct %lu (%f%%) ignore %lu\n", correct, count, correct * 100.0 / count, almost_correct, almost_correct * 100.0 / count, ignore);

	return 0;	
}

int main(int argc, const char *argv[])
{
	Model model(200);
	int n_workers = 4;

	auto distance = [&model]() {
		while (1) {
			std::string s;
			std::cout << "\nInput(:quit to break):";
			std::cin >> s;
			if (s == ":quit") break;
			auto p = model.most_similar(std::vector<std::string>{s}, std::vector<std::string>(), 10);
			size_t i = 0;
			for (auto& v: p) {
				std::cout << i++ << " " << v.first << " " << v.second << std::endl;
			}
		}
	};

	bool train = true, test = true;
	if (train) {
		std::vector<SentenceP> sentences;

		size_t count =0;
		const size_t max_sentence_len = 200;

		SentenceP sentence(new Sentence);
		// wget http://mattmahoney.net/dc/text8.zip
		std::ifstream in("text8");
		while (true) {
			std::string s;
			in >> s;
			if (s.empty()) break;	
			
			++count;
			sentence->tokens_.push_back(std::move(s));
			if (count == max_sentence_len) {
				count = 0;
				sentences.push_back(std::move(sentence));
				sentence.reset(new Sentence);
			}
		}
		
		if (!sentence->tokens_.empty())
			sentences.push_back(std::move(sentence));

		model.build_vocab(sentences);
		model.train(sentences, n_workers);
		model.save("vectors.txt");

//		distance();
	}

	if (test) {
		model.load("vectors.txt");
		distance();
		accuracy(model, "questions-words.txt");
	}


	return 0;
}

