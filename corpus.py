"""Legal text used by the game.

* ``LABELED_CLAUSES``  - (clause type, sentence) pairs. They are used for
  (a) training the n-gram next-word predictor and
  (b) fine-tuning GPT-2 to classify clause types (scripts/train_classifier.py).
* ``GAME_PROMPTS``     - the partial sentences shown to the player.

The text is written for teaching purposes. It is not legal advice.
"""

from __future__ import annotations

from typing import NamedTuple

LABELS = [
    "Confidentiality",
    "Termination",
    "Indemnification",
    "Governing Law",
    "Payment",
    "Limitation of Liability",
    "Force Majeure",
    "Assignment",
    "Warranty",
]

LABELED_CLAUSES: list[tuple[str, str]] = [
    # ---------------------------------------------------------------- Confidentiality
    ("Confidentiality", "The Receiving Party shall hold all Confidential Information in strict confidence and shall not disclose it to any third party without the prior written consent of the Disclosing Party."),
    ("Confidentiality", "Each party agrees to use the Confidential Information solely for the purpose of evaluating the proposed transaction."),
    ("Confidentiality", "Confidential Information does not include information that is or becomes publicly available through no fault of the Receiving Party."),
    ("Confidentiality", "The obligations of confidentiality under this Agreement shall survive for a period of five years following its termination."),
    ("Confidentiality", "The Receiving Party shall protect the Confidential Information using at least the same degree of care it uses to protect its own confidential information."),
    ("Confidentiality", "Upon written request, the Receiving Party shall promptly return or destroy all materials containing Confidential Information."),
    ("Confidentiality", "Neither party shall disclose the terms of this Agreement to any third party except to its professional advisers who are bound by duties of confidence."),
    ("Confidentiality", "The Receiving Party may disclose Confidential Information to the extent required by law or by order of a court of competent jurisdiction."),
    ("Confidentiality", "All proprietary information disclosed by one party to the other shall remain the exclusive property of the Disclosing Party."),
    ("Confidentiality", "The Receiving Party shall notify the Disclosing Party immediately upon discovery of any unauthorized use or disclosure of Confidential Information."),
    ("Confidentiality", "Each party shall restrict access to Confidential Information to those employees who have a need to know and who are bound by written confidentiality obligations."),
    ("Confidentiality", "Nothing in this Agreement grants the Receiving Party any licence or right in the Confidential Information other than the limited right to use it as expressly permitted."),
    # ---------------------------------------------------------------- Termination
    ("Termination", "Either party may terminate this Agreement at any time by giving thirty days prior written notice to the other party."),
    ("Termination", "This Agreement shall terminate automatically upon the expiration of the Initial Term unless renewed in writing by both parties."),
    ("Termination", "Either party may terminate this Agreement immediately upon written notice if the other party commits a material breach and fails to cure it within fourteen days."),
    ("Termination", "Upon termination of this Agreement, all rights and licences granted to the Licensee shall immediately cease."),
    ("Termination", "The Company may terminate this Agreement for cause if the Contractor becomes insolvent or files for bankruptcy."),
    ("Termination", "Termination of this Agreement shall not affect any rights or obligations that accrued before the effective date of termination."),
    ("Termination", "Upon expiration or termination, each party shall return all property belonging to the other party within ten business days."),
    ("Termination", "Either party may terminate this Agreement for convenience without liability by providing ninety days written notice."),
    ("Termination", "If either party materially breaches this Agreement, the non-breaching party may terminate it by written notice with immediate effect."),
    ("Termination", "The Employer may terminate the employment of the Employee without notice in the event of gross misconduct."),
    ("Termination", "Sections relating to confidentiality, indemnification and payment obligations shall survive the termination of this Agreement."),
    ("Termination", "This Agreement may be terminated by mutual written agreement of the parties at any time."),
    # ---------------------------------------------------------------- Indemnification
    ("Indemnification", "The Supplier shall indemnify and hold harmless the Customer from and against any claims, damages, losses and expenses arising out of the Supplier's breach of this Agreement."),
    ("Indemnification", "Each party shall defend, indemnify and hold harmless the other party against all third party claims arising from its negligence or wilful misconduct."),
    ("Indemnification", "The Indemnifying Party shall pay all costs, including reasonable legal fees, incurred by the Indemnified Party in connection with any claim covered by this section."),
    ("Indemnification", "The Indemnified Party shall give the Indemnifying Party prompt written notice of any claim for which indemnification is sought."),
    ("Indemnification", "The Licensor shall indemnify the Licensee against any claim that the Software infringes the intellectual property rights of a third party."),
    ("Indemnification", "The Indemnifying Party shall have sole control of the defence and settlement of any claim, provided that it shall not settle without the consent of the Indemnified Party."),
    ("Indemnification", "The Contractor agrees to indemnify the Client for any loss arising from personal injury or property damage caused by the Contractor's employees."),
    ("Indemnification", "The Tenant shall indemnify the Landlord against all liabilities resulting from the Tenant's use of the Premises."),
    ("Indemnification", "The obligation to indemnify under this clause shall apply regardless of whether the claim is groundless, false or fraudulent."),
    ("Indemnification", "The Seller shall hold the Buyer harmless from any liability arising from defects in the goods supplied under this Agreement."),
    ("Indemnification", "The Indemnified Party shall cooperate fully with the Indemnifying Party in the defence of any such claim at the Indemnifying Party's expense."),
    ("Indemnification", "The indemnity in this clause shall not apply to the extent that the loss was caused by the negligence of the Indemnified Party."),
    # ---------------------------------------------------------------- Governing Law
    ("Governing Law", "This Agreement shall be governed by and construed in accordance with the laws of the State of New York."),
    ("Governing Law", "This Agreement and any non-contractual obligations arising out of it shall be governed by the laws of England and Wales."),
    ("Governing Law", "The laws of the Republic of India shall govern the validity, interpretation and performance of this Agreement."),
    ("Governing Law", "This Agreement shall be interpreted under the laws of the State of California without regard to its conflict of law principles."),
    ("Governing Law", "The parties agree that the courts of Singapore shall have exclusive jurisdiction over any matter arising under this Agreement."),
    ("Governing Law", "The governing law of this Agreement shall be the law of Ontario and the federal laws of Canada applicable therein."),
    ("Governing Law", "Each party irrevocably submits to the exclusive jurisdiction of the courts located in Delaware for any action arising out of this Agreement."),
    ("Governing Law", "This Agreement shall be governed by the substantive laws of Germany, excluding the United Nations Convention on Contracts for the International Sale of Goods."),
    ("Governing Law", "The validity and enforcement of this Agreement shall be determined in accordance with the laws of the jurisdiction in which the Company is registered."),
    ("Governing Law", "Any legal action or proceeding relating to this Agreement shall be brought exclusively in the state or federal courts of Texas."),
    ("Governing Law", "This Agreement shall be construed in accordance with the laws of Karnataka, India, and the courts at Bengaluru shall have jurisdiction."),
    ("Governing Law", "The choice of law set out in this clause shall apply regardless of where the parties perform their obligations."),
    # ---------------------------------------------------------------- Payment
    ("Payment", "The Client shall pay the Contractor the fees set out in Schedule A within thirty days of receipt of a valid invoice."),
    ("Payment", "All invoices shall be paid in United States dollars by wire transfer to the account designated by the Supplier."),
    ("Payment", "Late payments shall accrue interest at the rate of one and a half percent per month until paid in full."),
    ("Payment", "The Customer shall pay all applicable taxes, duties and levies arising from the supply of the services."),
    ("Payment", "The fees are non-refundable and shall be paid in advance on the first day of each calendar month."),
    ("Payment", "The Buyer shall pay the purchase price in full on or before the closing date."),
    ("Payment", "If the Client disputes an invoice in good faith, it shall pay the undisputed portion and notify the Supplier of the dispute within ten days."),
    ("Payment", "The Company shall reimburse the Consultant for reasonable expenses incurred in the performance of the services, subject to prior written approval."),
    ("Payment", "Payment shall be made without set-off, deduction or counterclaim of any kind."),
    ("Payment", "The Tenant shall pay rent monthly in advance on or before the first day of each month."),
    ("Payment", "The Supplier may suspend the services if any undisputed amount remains unpaid for more than fifteen days after the due date."),
    ("Payment", "Total compensation under this Agreement shall not exceed the amount specified in the applicable statement of work."),
    # ---------------------------------------------------------------- Limitation of Liability
    ("Limitation of Liability", "In no event shall either party be liable to the other for any indirect, incidental, special or consequential damages."),
    ("Limitation of Liability", "The total aggregate liability of the Supplier under this Agreement shall not exceed the fees paid by the Customer in the preceding twelve months."),
    ("Limitation of Liability", "Nothing in this Agreement limits or excludes liability for death or personal injury caused by negligence or for fraud."),
    ("Limitation of Liability", "The Licensor shall not be liable for any loss of profits, loss of revenue or loss of data arising out of the use of the Software."),
    ("Limitation of Liability", "Except as expressly provided in this Agreement, the Provider excludes all liability for any loss or damage however caused."),
    ("Limitation of Liability", "The limitations of liability in this section shall apply even if the party has been advised of the possibility of such damages."),
    ("Limitation of Liability", "Under no circumstances shall the Company's liability for any claim exceed the total amount paid under this Agreement."),
    ("Limitation of Liability", "Neither party shall be liable for any loss of goodwill or reputation arising from the performance of this Agreement."),
    ("Limitation of Liability", "The Service Provider's maximum liability arising from any breach shall be limited to the direct damages actually suffered by the Client."),
    ("Limitation of Liability", "These limitations shall apply to the fullest extent permitted by applicable law."),
    ("Limitation of Liability", "The Contractor shall have no liability for any damage resulting from information supplied by the Client that is inaccurate or incomplete."),
    ("Limitation of Liability", "The exclusions and caps in this clause are agreed to be a reasonable allocation of risk between the parties."),
    # ---------------------------------------------------------------- Force Majeure
    ("Force Majeure", "Neither party shall be liable for any failure or delay in performance caused by events beyond its reasonable control."),
    ("Force Majeure", "Force Majeure Events include acts of God, war, terrorism, riots, fire, flood, earthquake, epidemic and governmental action."),
    ("Force Majeure", "The affected party shall notify the other party promptly in writing of the occurrence of a Force Majeure Event."),
    ("Force Majeure", "If a Force Majeure Event continues for more than sixty days, the parties shall meet to discuss in good faith how to proceed."),
    ("Force Majeure", "The party affected by a Force Majeure Event shall use reasonable efforts to mitigate its effects and resume performance as soon as possible."),
    ("Force Majeure", "A lack of funds shall not constitute a Force Majeure Event under this Agreement."),
    ("Force Majeure", "The time for performance shall be extended by a period equal to the delay caused by the Force Majeure Event."),
    ("Force Majeure", "Obligations to make payments that have already fallen due shall not be excused by a Force Majeure Event."),
    ("Force Majeure", "Neither party shall be deemed in breach of this Agreement as a result of delays caused by natural disasters, pandemics or acts of government."),
    ("Force Majeure", "The Supplier shall not be responsible for any failure to deliver goods where the failure results from an unforeseeable event outside its control."),
    ("Force Majeure", "Performance of the affected obligations shall be suspended for as long as the Force Majeure Event prevents compliance."),
    ("Force Majeure", "Any event that could have been prevented by the exercise of reasonable diligence shall not qualify as a Force Majeure Event."),
    # ---------------------------------------------------------------- Assignment
    ("Assignment", "Neither party may assign or transfer any of its rights or obligations under this Agreement without the prior written consent of the other party."),
    ("Assignment", "Any purported assignment in violation of this section shall be null and void."),
    ("Assignment", "This Agreement shall be binding upon and inure to the benefit of the parties and their respective successors and permitted assigns."),
    ("Assignment", "The Licensee may not sublicense, assign or otherwise transfer the licence granted under this Agreement to any third party."),
    ("Assignment", "Either party may assign this Agreement to an affiliate or to a successor in connection with a merger or sale of substantially all of its assets."),
    ("Assignment", "The Contractor shall not subcontract any of its obligations under this Agreement without the written approval of the Client."),
    ("Assignment", "A change of control of the Supplier shall be deemed an assignment requiring the consent of the Customer."),
    ("Assignment", "The Assignor shall remain jointly liable with the Assignee for the performance of all obligations assigned under this Agreement."),
    ("Assignment", "Consent to assignment shall not be unreasonably withheld, conditioned or delayed."),
    ("Assignment", "The Company may assign its rights under this Agreement to any entity that acquires all or substantially all of its business."),
    ("Assignment", "No assignment shall relieve the assigning party of its obligations accrued before the date of assignment."),
    ("Assignment", "The Employee may not delegate the performance of the duties under this Agreement to any other person."),
    # ---------------------------------------------------------------- Warranty
    ("Warranty", "The Supplier warrants that the goods shall be free from defects in materials and workmanship for a period of twelve months from delivery."),
    ("Warranty", "Each party represents and warrants that it has full power and authority to enter into and perform this Agreement."),
    ("Warranty", "The Provider warrants that the services shall be performed with reasonable skill and care in accordance with industry standards."),
    ("Warranty", "The Licensor warrants that it owns all rights necessary to grant the licence set out in this Agreement."),
    ("Warranty", "Except as expressly stated in this Agreement, all warranties, conditions and representations, whether express or implied, are excluded."),
    ("Warranty", "The Software is provided as is without warranty of any kind, including warranties of merchantability or fitness for a particular purpose."),
    ("Warranty", "The Seller warrants that it has good and marketable title to the goods and that they are free from any lien or encumbrance."),
    ("Warranty", "In the event of a breach of warranty, the Customer's sole remedy shall be repair or replacement of the defective goods."),
    ("Warranty", "The Contractor warrants that the work shall comply with all applicable laws, regulations and building codes."),
    ("Warranty", "Each party represents that the execution of this Agreement does not violate any other agreement to which it is bound."),
    ("Warranty", "The warranties in this section shall not apply to defects caused by misuse, unauthorised modification or normal wear and tear."),
    ("Warranty", "The Consultant warrants that the deliverables shall not infringe the intellectual property rights of any third party."),
]

# Extra unlabeled sentences that only enrich the n-gram corpus.
EXTRA_SENTENCES: list[str] = [
    "This Agreement constitutes the entire agreement between the parties and supersedes all prior negotiations and understandings.",
    "No amendment to this Agreement shall be effective unless it is in writing and signed by both parties.",
    "If any provision of this Agreement is held to be invalid, the remaining provisions shall continue in full force and effect.",
    "Any notice under this Agreement shall be in writing and delivered by hand or by registered post to the address of the other party.",
    "The failure of either party to enforce any right under this Agreement shall not constitute a waiver of that right.",
    "The parties are independent contractors and nothing in this Agreement creates a partnership or agency between them.",
]


class GamePrompt(NamedTuple):
    clause_type: str
    starter: str


GAME_PROMPTS: list[GamePrompt] = [
    GamePrompt("Confidentiality", "The Receiving Party shall"),
    GamePrompt("Confidentiality", "Each party agrees to"),
    GamePrompt("Termination", "Either party may terminate this Agreement"),
    GamePrompt("Termination", "Upon termination of this Agreement,"),
    GamePrompt("Indemnification", "The Supplier shall indemnify"),
    GamePrompt("Governing Law", "This Agreement shall be governed by"),
    GamePrompt("Payment", "The Client shall pay"),
    GamePrompt("Limitation of Liability", "In no event shall"),
    GamePrompt("Force Majeure", "Neither party shall be liable for"),
    GamePrompt("Assignment", "Neither party may assign"),
    GamePrompt("Warranty", "The Supplier warrants that"),
    GamePrompt("Warranty", "Each party represents and warrants that"),
]


def all_sentences() -> list[str]:
    """Every sentence available for n-gram training."""
    return [s for _, s in LABELED_CLAUSES] + EXTRA_SENTENCES
