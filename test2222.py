def evaluate_tree(tree):
        # Searching for cause-effect relations that involve a verb/action we look for NP-VP-NP
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP' and t.right_sibling() is not None):
            sibling = subtree.right_sibling()

            # skip to the first verb
            while sibling.label() == 'ADVP' and sibling.right_sibling() is not None:
                sibling = sibling.right_sibling()

            # NP-VP-NP pattern found .__repr__()
            if sibling.label() == 'VP' and "('NP'" in sibling.__repr__():
                np_string1 = ' '.join([p[0]['nlpToken']['originalText'] for p in subtree.pos()])
                np_string2 = ' '.join([p[0]['nlpToken']['originalText'] for p in sibling.pos()])
                who_what = np_string1 +' '+ np_string2
                return who_what
