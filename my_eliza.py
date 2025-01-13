import logging
import random
import re
from collections import namedtuple

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

    def load(self, path):
        key = None
        decomp = None
        with open(path) as file:
            for line in file:
                if not line.strip():
                    continue
                tag, content = [part.strip() for part in line.split(':')]
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
                index = int(reword[1:-1])   # gets the number of the matched word from user input
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
                # Store the key phrase as a properly joined string
                key_phrase = ' '.join(words).strip()
                self.memory_keys.append(key_phrase)
                self.memory.append(output)
                log.debug('Saved to memory_keys: %s', key_phrase)
                continue
            return output
        return None

    def respond(self, text):
        # checks for quit commands
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
            "Earlier you said '{}'. How does that affect the situation?",
            "You spoke before about '{}'. Tell me more about that.",
            "Let's return to '{}'. How does this relate to your current thoughts?"
        ]

        # check for yes/no responses, and use items from memory to further conversation
        if any(word.lower() in ['yes', 'no', 'yeah', 'yep', 'y', 'n'] for word in words):
            if self.memory_keys and self.memory:  # Check both lists have items
                index = random.randrange(len(self.memory_keys))
                memory_item = self.memory_keys.pop(index).strip()
                self.memory.pop(index)
                prompt = random.choice(memory_prompts)
                return prompt.format(memory_item)

        # applies pre-substitutions (converts contractions and common phrases)
        words = self._sub(words, self.pres)
        log.debug('After pre-substitution: %s', words)

        # finds matching keywords and sorts them by weight
        keys = [self.keys[w.lower()] for w in words if w.lower() in self.keys]
        keys = sorted(keys, key=lambda k: -k.weight)
        log.debug('Sorted keys: %s', [(k.word, k.weight) for k in keys])

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
                # uses a saved response from memory?
                index = random.randrange(len(self.memory))
                output = self.memory.pop(index)
                log.debug('Output from memory: %s', output)
            else:
                # default response if there are responses from memory
                output = self._next_reasmb(self.keys['xnone'].decomps[0])
                log.debug('Output from xnone: %s', output)

        return " ".join(output)

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
    print('Memory: ', eliza.memory)


if __name__ == '__main__':
    logging.basicConfig()
    main()
