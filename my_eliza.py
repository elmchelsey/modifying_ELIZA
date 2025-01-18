import logging
import random
import re
from collections import namedtuple
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk import RegexpParser
import spacy

nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Fix Python2/Python3 incompatibility
try: input = raw_input
except NameError: pass

log = logging.getLogger(__name__)


class Key:
    def __init__(self, word, weight, decomps):
        self.word = word
        self.weight = weight
        self.decomps = decomps


class Decomp:
    def __init__(self, parts, save, reasmbs):
        self.parts = parts
        self.save = save
        self.reasmbs = reasmbs
        self.next_reasmb_index = 0


class Eliza:
    def __init__(self):
        self.initials = []
        self.finals = []
        self.quits = []
        self.pres = {}
        self.posts = {}
        self.synons = {}
        self.keys = {}
        self.memory = []    # stores entire responses from memory based on keywords in prev inputs
        self.memory_keys = []   # stores keywords only

        self.suicide_keywords = ['suicide', 'suicidal', 'don\'t want to live', 'kill myself', 'want to die', 'want to kill myself', 'want to die', 'want to kill myself', 'kms']

        self.sia = SentimentIntensityAnalyzer()

        self.sentiment_responses = {
            'very_neg': [
                'I hear that you are feeling very upset about this.',
                'This seems to be causing you a lot of distress.',
                'I know that this is a difficult time for you, and I want to help.',
            ],
            'neg': [
                'That sounds really tough.',
                'I am sorry to hear that you are feeling this way.',
            ],
            'neutral': [
                'I see.',
                'Hmm.',
                'I hear you.',
            ],
            'pos': [
                'I am glad to hear that.',
                'That is amazing to hear.',
            ],
            'very_pos': [
                'That is really awesome to hear!',
                'I am really glad to hear that you are feeling this way.',
            ]
        }

    def load(self, path):
        key = None
        decomp = None
        with open(path) as file:
            for line in file:
                line = line.strip()
                # Skip empty lines or lines without a colon
                if not line or ':' not in line:
                    continue
                tag, content = [part.strip() for part in line.split(':', 1)]
                if tag == 'initial':
                    self.initials.append(content)
                elif tag == 'final':
                    self.finals.append(content)
                elif tag == 'quit':
                    self.quits.append(content)
                elif tag == 'pre':
                    parts = content.split(' ')
                    self.pres[parts[0]] = parts[1:]
                elif tag == 'post':
                    parts = content.split(' ')
                    self.posts[parts[0]] = parts[1:]
                elif tag == 'synon':
                    parts = content.split(' ')
                    self.synons[parts[0]] = parts
                elif tag == 'key':
                    parts = content.split(' ')
                    word = parts[0]
                    weight = int(parts[1]) if len(parts) > 1 else 1
                    key = Key(word, weight, [])
                    self.keys[word] = key
                elif tag == 'decomp':
                    parts = content.split(' ')
                    save = False
                    if parts[0] == '$':
                        save = True
                        parts = parts[1:]
                    decomp = Decomp(parts, save, [])
                    key.decomps.append(decomp)
                elif tag == 'reasmb':
                    parts = content.split(' ')
                    decomp.reasmbs.append(parts)

    def _match_decomp_r(self, parts, words, results):
        if not parts and not words:
            return True
        if not parts or (not words and parts != ['*']):
            return False
        if parts[0] == '*':
            for index in range(len(words), -1, -1):
                results.append(words[:index])   # captures matched words from user input
                if self._match_decomp_r(parts[1:], words[index:], results):
                    return True
                results.pop()
            return False
        elif parts[0].startswith('@'):
            root = parts[0][1:]
            if not root in self.synons:
                raise ValueError("Unknown synonym root {}".format(root))
            if not words[0].lower() in self.synons[root]:
                return False
            results.append([words[0]])
            return self._match_decomp_r(parts[1:], words[1:], results)
        elif parts[0].lower() != words[0].lower():
            return False
        else:
            return self._match_decomp_r(parts[1:], words[1:], results)

    def _match_decomp(self, parts, words):
        results = []
        if self._match_decomp_r(parts, words, results):
            return results
        return None

    def _next_reasmb(self, decomp):
        index = decomp.next_reasmb_index
        result = decomp.reasmbs[index % len(decomp.reasmbs)]
        decomp.next_reasmb_index = index + 1
        return result

    def _reassemble(self, reasmb, results):
        output = []
        for reword in reasmb:
            if not reword:
                continue
            if reword[0] == '(' and reword[-1] == ')':
                index = int(reword[1:-1])   # gets the number of the matched wordss from user input
                if index < 1 or index > len(results):
                    raise ValueError("Invalid result index {}".format(index))
                insert = results[index - 1] # uses the captured text from user input
                for punct in [',', '.', ';']:
                    if punct in insert:
                        insert = insert[:insert.index(punct)]
                output.extend(insert)
            else:
                output.append(reword)
        return output

    def _sub(self, words, sub):
        output = []
        for word in words:
            word_lower = word.lower()
            if word_lower in sub:
                output.extend(sub[word_lower])
            else:
                output.append(word)
        return output


    # matches key words and sorts them by weight
    def _match_key(self, words, key):
        for decomp in key.decomps:
            results = self._match_decomp(decomp.parts, words)
            if results is None:
                log.debug('Decomp did not match: %s', decomp.parts)
                continue
            
            results = [self._sub(words, self.posts) for words in results]
            reasmb = self._next_reasmb(decomp)
            
            if reasmb[0] == 'goto':
                goto_key = reasmb[1]
                if not goto_key in self.keys:
                    raise ValueError("Invalid goto key {}".format(goto_key))
                return self._match_key(words, self.keys[goto_key])
            
            output = self._reassemble(reasmb, results)
            if decomp.save:
                # Store the complete response
                response = ' '.join(output)
                # Store the original input
                key_phrase = ' '.join(words)
                
                # Only store if not empty
                if key_phrase.strip() and response.strip():
                    self.memory_keys.append(key_phrase)
                    self.memory.append(response)
                    log.debug('Saved to memory - Key: %s, Response: %s', key_phrase, response)
            return output
        return None
    
    def _handle_crisis(self):

        crisis_resources = [
            "National Suicide Prevention Lifeline: 988 or 1-800-273-8255 \n",
            "Crisis Text Line: Text HOME to 741741 \n"
        ]
        
        crisis_topics = [
            'Self Description',
            'Intensity Report',
            'Duration Report',
            'Plan'
        ]

        crisis_questions = [
            'I hear that you are in pain. Can you tell me a bit about the suicidal thoughts?',
            'How intense are these thoughts?',
            'How long do the thoughts last?',
            'Have you made a plan?',
        ]
        
        responses = []

        for question in crisis_questions:
            response = input(question + ' > ')
            responses.append(response)

        print('Thank you for sharing this with me. Please know that you are not alone, and that there are resources available to you. \\')
        print('Here are some resources: \n\n')
        print('----------------------------------')
        for resource in crisis_resources:
            print(resource)
        print('----------------------------------')
        print('I have compiled your responses, and I recommend you send this report to a mental health professional in your area to receive specialized support.\n')

        with open('suicide_responses.txt', 'w') as file:
            file.write('Suicide Risk Report\n')
            for topic, response in zip(crisis_topics, responses):
                file.write(f"{topic}: {response}\n")

            # SAMHSA considers a high risk if the person has a plan
            if responses[3] == 'yes':
                file.write('Suicide Risk: High\n')

            # SAMHSA considers a high risk if the person declines to answer screening questions
            if 'no' in responses or 'No' in responses:
                file.write('Declined to answer screening questions, be advised.')

    def _get_sentiment_based_response(self, text):
        scores = self.sia.polarity_scores(text)    # NLKT sentiment analysis
        compound_score = scores['compound']

        # VADER compound score ranges from -1 (very negative) to +1 (very positive)
        if compound_score <= -0.5:
            return random.choice(self.sentiment_responses['very_neg'])
        elif compound_score <= -0.1:
            return random.choice(self.sentiment_responses['neg'])
        elif compound_score >= 0.5:
            return random.choice(self.sentiment_responses['very_pos'])
        elif compound_score >= 0.1:
            return random.choice(self.sentiment_responses['pos'])
        elif text == 'yes' or text == 'no':
            return random.choice(self.sentiment_responses['neutral'])
        else:
            return random.choice(self.sentiment_responses['neutral'])

    def respond(self, text):
        # Add early return for repeated one-word responses
        if text.lower() in ['yes', 'no'] and hasattr(self, 'last_input') and self.last_input == text.lower():
            return "I notice you're repeating yourself. Would you like to tell me more about what's on your mind?"
        
        self.last_input = text.lower()

        # Existing suicide check
        if any(keyword in text.lower() for keyword in self.suicide_keywords):
            return self._handle_crisis()

        if text.lower() in self.quits:
            return None

        # cleans up punctuation
        text = re.sub(r'\s*\.+\s*', ' . ', text)
        text = re.sub(r'\s*,+\s*', ' , ', text)
        text = re.sub(r'\s*;+\s*', ' ; ', text)
        log.debug('After punctuation cleanup: %s', text)

        # splits into words
        words = [w for w in text.split(' ') if w]
        log.debug('Input: %s', words)

        memory_prompts = [
            "You spoke before about '{}'. Tell me more about that.",
            "Let's return to '{}'. How does this relate to your current thoughts?"
        ]

        # check for single-word yes/no responses, and uses items from memory if there are any
        if len(words) == 1 and words[0].lower() in ['yes', 'no']:
            if self.memory_keys:
                index = random.randrange(len(self.memory_keys))
                memory_item = str(self.memory_keys[index])
                memory_item = ' '.join(memory_item.split())
                self.memory_keys.pop(index)
                prompt = random.choice(memory_prompts)
                return prompt.format(memory_item)
            else:
                words = words

        # applies pre-substitutions (converts contractions and common phrases)
        words = self._sub(words, self.pres)
        log.debug('After pre-substitution: %s', words)

        #define a noun phrase: optional determiner, optional adjective(s), and a noun
        noun_phrase = 'NP: {<DT>?<JJ>*<NN.*>+}'
        chunk_parser = nltk.RegexpParser(noun_phrase)

        # parse the pos tagged inputs
        pos_tags = pos_tag(words)
        tree = chunk_parser.parse(pos_tags)

        noun_phrases = []
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            phrase = ' '.join(word for word, tag in subtree.leaves())

            if len(phrase.split()) > 1 and (len(phrase) > 3 and 
                                           not phrase.lower() in ['the', 'this', 'that', 'she', 'he', 'it', 'we', 'you', 'they', 'i']):
                noun_phrases.append(phrase)

            self.memory_keys.extend(noun_phrases)

        # finds matching keywords and sorts them by weight
        keys = [self.keys[w.lower()] for w in words if w.lower() in self.keys]
        keys = sorted(keys, key=lambda k: -k.weight)
        log.debug('Sorted keys: %s', [(k.word, k.weight) for k in keys])
        log.debug('Sorted keys saved to memory keys: %s', self.memory_keys)
        log.debug('Current memory keys: %s', self.memory_keys)
 
        output = None

        # generates a response using key words
        for key in keys:
            output = self._match_key(words, key)
            if output:
                log.debug('Output from key: %s', output)
                break

        # fallback responses if there are no key words
        if not output:
            if self.memory:
                # uses a saved response from memory
                index = random.randrange(len(self.memory))
                output = self.memory.pop(index)
                log.debug('Output from memory: %s', output)
            else:
                # default response if there are responses from memory
                output = self._next_reasmb(self.keys['xnone'].decomps[0])
                log.debug('Output from xnone: %s', output)

        if output:
            final_response = []

            # Increase sentiment response probability for longer inputs
            if len(words) > 3 and random.random() < 0.4:
                sentiment_response = self._get_sentiment_based_response(text)
                final_response.extend(sentiment_response.split())

            final_response.extend(output)
            return " ".join(final_response)
        
        return " ".join(output) if output else "Could you tell me more about that?"  # Add default fallback

    def initial(self):
        return random.choice(self.initials)

    def final(self):
        return random.choice(self.finals)

    def run(self):
        print(self.initial())

        while True:
            sent = input('> ')

            output = self.respond(sent)
            if output is None:
                break

            print(output)

        print(self.final())


def main():
    eliza = Eliza()
    eliza.load('my_doctor.txt')
    eliza.run()


if __name__ == '__main__':
    logging.basicConfig()
    main()
