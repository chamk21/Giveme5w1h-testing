import logging

from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from nltk.tree import ParentedTree
import re
"""
This is a simple example how to use the extractor in combination with a dict in news-please format.

- Nothing is cached

"""

# don`t forget to start up core_nlp_host
# giveme5w1h-corenlp

titleshort = "Barack Obama was born in Hawaii.  He is the president. Obama was elected in 2008."

title = "Taliban attacks German consulate in northern Afghan city of Mazar-i-Sharif with truck bomb"
lead = "The death toll from a powerful Taliban truck bombing at the German consulate in Afghanistan's Mazar-i-Sharif city rose to at least six Friday, with more than 100 others wounded in a major militant assault."
text5 = '''Aug 2 - Canada's Toronto Dominion Bank (TD.TO) said it will buy New York-based boutique investment bank Cowen Inc (COWN.O) for $1.3 billion in cash, seeking to boost its presence in the high-growth U.S market.The deal marks TD's second acquisition bid in the United States this year and the Canada's second-largest lender by market value has made no secret of its ambitions to expand in the world's biggest economy. TD will fund the acquisition from the $1.9 billion proceeds from the sale of shares of Charles Schwab , announced on Monday.
Canada's major banks have been on a shopping spree south of the border in the past year, seeking growth away from home, where the Big Six banks control nearly 90% of the market. read more
National Bank Financial analysts said the deal provides "valuable diversification" of TD's U.S. capital markets business, but flagged integration as a primary risk, saying it is "notoriously difficult when involving investment banking operations with different cultures."
In February, TD said it would buy Memphis-based First Horizon Corp (FHN.N) for $13.4 billion in its biggest ever acquisition. read more 
Investors had already expressed concern around integration following the First Horizon deal. read more
The Cowen deal values the target at $39 a share, a nearly 10% premium to its last closing price. Cowen shares rose 8.9% in morning trading in New York. TD shares slipped 1.3%.
The transaction is "modestly positive (for TD), especially given that it is on-strategy with the bank’s push into its U.S.-dollar platform," Credit Suisse Analyst Joo Ho Kim wrote in a note.
On Monday, TD said it was selling 28.4 million shares of Schwab, reducing its ownership to about 12% from 13.4%. That stake was the result of Schwab's purchase of TD Ameritrade, of which TD owned 43%. TD said it has no current plans to sell more Schwab shares.
TD expects pre-tax integration costs of about $450 million over three years, and revenue synergies of $300-350 million by the third year. TD must pay a termination fee of $42.25 million if it cancels the deal because of a recommendation change or another superior proposal.
The deal is expected to close in the first quarter of 2023.'''

text6 = '''Susan Bannigan has been appointed as the new Board Chair of Milk Crate Theatre.

Bannigan has joined Milk Crate Theatre having recently moved on from her position as CEO of the Westpac Foundation and Westpac Scholarship Trust where she worked closely with Westpac, community groups, social entrepreneurs and the business sector to support new innovations in addressing the complex issues of homelessness, long-term unemployment, social inclusion for refugees and those living with issues of mental health in communities across Australia.

She has over 30 years’ experience in experience in the financial services and philanthropic industries in Europe, Pacific and Australia. Bannigan’s former Board roles include Chair of the Business/Higher Education Round Table and Director of Variety NSW. She is a Chartered Accountant, member of the Australian Institute of Company Directors and holds a Bachelor’s degree in Economics from the University of Sydney.

Milk Crate Theatre CEO, Jodie Wainwright, said: ‘She brings extensive leadership and governance experience from her distinguished career in banking and with the Westpac Foundation. Susan has been a long-term supporter and her skills will be of immense value to Milk Crate Theatre. The board and team look forward to working with Susan to realise our vision of effecting social change through the power of performance,’ Wainwright said.

Bannigan will replace Michael Sirmai, who has been a member of the Milk Crate Board since 2013 and Board Chair since 2016.

‘Under Michael’s stewardship we have been able to navigate many of the challenges that faced small to medium arts organisations and come through the COVID pandemic in a strong position, poised for growth,’ said Wainwright. 

‘During his period as Chair, we have created a new and sustainable business plan and structure. We have also invested in our Social Impact Framework, with a robust Theory of Change and have embedded impact measurement into the program design. We thank Michael for his leadership and wish him well in his future endeavours,’ she added.

Michael Sirmai added: ‘I am absolutely delighted for someone of Susan’s experience and reputation to lead the Board over the company’s next phase.’

Bannigan commenced her role as Board Chair effective 15 August.'''



date_publish = '2016-11-10 07:44:00'

minimal_length_of_tokens = 3

if __name__ == '__main__':
    # logger setup
    log = logging.getLogger('GiveMe5W')
    log.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    log.addHandler(sh)

    # giveme5w setup - with defaults
    extractor = MasterExtractor()
    doc = Document.from_text(text6, date_publish)

    doc = extractor.parse(doc)

    def cut_what(tree, min_length=0, length=0):
        if type(tree[0]) is not ParentedTree:
            # we found a leaf
            return ParentedTree(tree.label(), [tree[0]])
        else:
            children = []
            for sub in tree:
                child = cut_what(sub, min_length, length)
                length += len(child.leaves())
                children.append(child)
                if sub.label() == 'NP':
                    sibling = sub.right_sibling()
                    if length < min_length and sibling is not None and sibling.label() == 'PP':
                        children.append(sibling.copy(deep=True))
                    break
            return ParentedTree(tree.label(), children)


    def evaluate_tree(sentence_root):
        
        candidates = []
        for subtree in sentence_root.subtrees():
            if subtree.label() == 'NP' and subtree.parent().label() == 'S':

                # Skip NPs containing a VP
                if any(list(subtree.subtrees(filter=lambda t: t.label() == 'VP'))):
                    continue

                # check siblings for VP
                sibling = subtree.right_sibling()
                while sibling is not None:
                    if sibling.label() == 'VP':
                        # this gives a tuple to find the way from sentence to leaf
                        # tree_position = subtree.leaf_treeposition(0)
                        entry = [subtree.pos(), cut_what(sibling, minimal_length_of_tokens).pos(),
                                 sentence_root.stanfordCoreNLPResult['index']]
                        candidates.append(entry)
                        break
                    sibling = sibling.right_sibling()
        return candidates

    corefs = doc.get_corefs()
    trees = doc.get_trees()
    candidates = []

    for cluster in corefs:
        for mention in corefs[cluster]:
            for pattern in evaluate_tree(trees[mention['sentNum'] - 1]):
                np_string = ' '.join([p[0]['nlpToken']['originalText'] for p in pattern[0]])
                if re.sub(r'\s+', '', mention['text']) in np_string:
                    p0 = ' '.join([p[0]['nlpToken']['originalText'] for p in pattern[0]])
                    p1 = ' '.join([p[0]['nlpToken']['originalText'] for p in pattern[1]])
                    print("pattern[0]: ",p0,"=========","pattern[1]: ",p1,'\n')

