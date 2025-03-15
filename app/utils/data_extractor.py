import re
from typing import Dict, List, Tuple
from datetime import datetime
from decimal import Decimal
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from app.models import Invoice, Vendor, Address, InvoiceItem
from app.config import settings
import usaddress
import pycountry
import dateparser
from price_parser import Price

class DataExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_matchers()

    def _setup_matchers(self):
        self.matcher.add("INVOICE_NUMBER", [[{"LOWER": "invoice"}, {"LOWER": "number"}, {"IS_ASCII": True, "LENGTH": {">=": 5}}]])
        self.matcher.add("INVOICE_DATE", [[{"LOWER": "date"}, {"IS_ASCII": True, "LENGTH": {">=": 8, "<=": 10}}]])
        self.matcher.add("TOTAL_AMOUNT", [[{"LOWER": {"IN": ["total", "amount", "sum"]}}, {"LIKE_NUM": True}]])
        self.matcher.add("TAX_AMOUNT", [[{"LOWER": {"IN": ["tax", "vat", "gst"]}}, {"LIKE_NUM": True}]])

    def extract_data(self, ocr_result: Dict) -> Invoice:
        text = " ".join(ocr_result["words"])
        doc = self.nlp(text)

        invoice_number = self._extract_invoice_number(doc)
        vendor = self._extract_vendor(doc)
        invoice_date = self._extract_date(doc)
        grand_total, taxes, final_total = self._extract_totals(doc)
        items = self._extract_items(doc, ocr_result["boxes"])

        return Invoice(
            filename=ocr_result.get("filename", ""),
            invoice_number=invoice_number,
            vendor=vendor,
            invoice_date=invoice_date,
            grand_total=grand_total,
            taxes=taxes,
            final_total=final_total,
            items=items,
            pages=ocr_result.get("pages", 1)
        )

    def _extract_invoice_number(self, doc: Doc) -> str:
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "INVOICE_NUMBER":
                return doc[end-1].text
        # Fallback: look for any alphanumeric string that looks like an invoice number
        for token in doc:
            if re.match(r'^[A-Za-z0-9-]{5,}$', token.text):
                return token.text
        return ""

    def _extract_vendor(self, doc: Doc) -> Vendor:
        org_names = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        if org_names:
            vendor_name = max(org_names, key=len)  # Choose the longest organization name
        else:
            vendor_name = "Unknown"
        
        address = self._extract_address(doc)
        return Vendor(name=vendor_name, address=address)

    def _extract_address(self, doc: Doc) -> Address:
        address_text = " ".join([ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]])
        parsed_address, _ = usaddress.tag(address_text)
        
        street = parsed_address.get('AddressNumber', '') + ' ' + parsed_address.get('StreetName', '')
        city = parsed_address.get('PlaceName', '')
        state = parsed_address.get('StateName', '')
        postal_code = parsed_address.get('ZipCode', '')
        
        country_name = parsed_address.get('CountryName', '')
        country = pycountry.countries.search_fuzzy(country_name)[0].alpha_2 if country_name else ''

        return Address(
            street=street.strip(),
            city=city,
            state=state,
            country=country,
            postal_code=postal_code
        )

    def _extract_date(self, doc: Doc) -> datetime:
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "INVOICE_DATE":
                date_str = doc[end-1].text
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date
        
        # Fallback: look for any date-like string
        for token in doc:
            parsed_date = dateparser.parse(token.text)
            if parsed_date:
                return parsed_date
        
        return datetime.now()  # Default to current date if not found

    def _extract_totals(self, doc: Doc) -> Tuple[Decimal, Decimal, Decimal]:
        grand_total = Decimal('0.00')
        taxes = Decimal('0.00')
        final_total = Decimal('0.00')

        matches = self.matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "TOTAL_AMOUNT":
                price = Price.fromstring(doc[end-1].text)
                if price.amount:
                    final_total = Decimal(str(price.amount))
            elif self.nlp.vocab.strings[match_id] == "TAX_AMOUNT":
                price = Price.fromstring(doc[end-1].text)
                if price.amount:
                    taxes = Decimal(str(price.amount))

        grand_total = final_total - taxes
        return grand_total, taxes, final_total

    def _extract_items(self, doc: Doc, boxes: List[List[int]]) -> List[InvoiceItem]:
        items = []
        for i, sent in enumerate(doc.sents):
            if any(token.like_num for token in sent):
                description = " ".join([token.text for token in sent if not token.like_num and not token.is_currency])
                numbers = [Price.fromstring(token.text) for token in sent if token.like_num or token.is_currency]
                if len(numbers) >= 3:
                    quantity = int(numbers[0].amount) if numbers[0].amount else 1
                    unit_price = Decimal(str(numbers[1].amount)) if numbers[1].amount else Decimal('0.00')
                    total = Decimal(str(numbers[2].amount)) if numbers[2].amount else Decimal('0.00')
                    
                    # Use bounding box information to determine if this is likely a line item
                    if i < len(boxes) and (boxes[i][2] - boxes[i][0]) > (doc[0].doc.page.width * 0.5):
                        items.append(InvoiceItem(
                            description=description,
                            quantity=quantity,
                            unit_price=unit_price,
                            total=total
                        ))
        return items

data_extractor = DataExtractor()
