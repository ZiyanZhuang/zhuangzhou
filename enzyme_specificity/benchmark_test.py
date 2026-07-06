#!/usr/bin/env python
"""
Enzyme Specificity Model Benchmark
====================================
20 enzyme-substrate pairs (14 positive, 6 negative) across diverse catalytic types.
Sequences from UniProt, SMILES from PubChem.
"""

import json
import sys
import os
import time

# ── Test Cases ──────────────────────────────────────────────────────────────

# CYP sequences (UniProt)
CYP3A4 = "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELVAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKPVVLKVESRDGTVSGA"
CYP2D6 = "MGLEALVPLAVIVAIFLLLVDLMHRRQRWAARYPPGPLPLPGLGNLLHVDFQNTPYCFDQLRRRFGDVFSLQLAWTPVVVLNGLAAVREALVTHGEDTADRPPVPITQILGFGPRSQGVFLARYGPAWREQRRFSVSTLRNLGLGKKSLEQWVTEEAACLCAAFANHSGRPFRPNGLLDKAVSNVIASLTCGRRFEYDDPRFLRLLDLAQEGLKEESGFLREVLNAVPVLLHIPALAGKVLRFQKAFLTQLDELLTEHRMTWDPAQPPRDLTEAFLAEMEKAKGNPESSFNDENLRIVVADLFSAGMVTTSTTLAWGLLLMILHPDVQRRVQQEIDDVIGQVRRPEMGDQAHMPYTTAVIHEVQRFGDIVPLGVTHMTSRDIEVQGFRIPKGTTLITNLSSVLKDEAVWEKPFRFHPEHFLDAQGHFVKPEAFLPFSAGRRACLGEPLARMELFLFFTSLLQHFSFSVPTGQPRPSHHGVFAFLVSPSPYELCAVPR"
CYP1A2 = "MALSQSVPFSATELLLASAIFCLVFWVLKGLRPRVPKGLKSPPEPWGWPLLGHVLTLGKNPHLALSRMSQRYGDVLQIRIGSTPVLVLSRLDTIRQALVRQGDDFKGRPDLYTSTLITDGQSLTFSTDSGPVWAARRRLAQNALNTFSIASDPASSSSCYLEEHVSKEAKALISRLQELMAGPGHFDPYNQVVVSVANVIGAMCFGQHFPESSDEMLSLVKNTHEFVETASSGNPLDFFPILRYLPNPALQRFKAFNQRFLWFLQKTVQEHYQDFDKNSVRDITGALFKHSKKGPRASGNLIPQEKIVNLVNDIFGAGFDTVTTAISWSLMYLVTKPEIQRKIQKELDTVIGRERRPRLSDRPQLPYLEAFILETFRHSSFLPFTIPHSTTRDTTLNGFYIPKKCCVFVNQWQVNHDPELWEDPSEFRPERFLTADGTAINKPLSEKMMLFGMGKRRCIGEVLAKWEIFLFLAILLQQLEFSVPPGVKVDLTPIYGLTMKHARCEHVQARLRFSIN"
CYP2C9 = "MDSLVVLVLCLSCLLLLSLWRQSSGRGKLPPGPTPLPVIGNILQIGIKDISKSLTNLSKVYGPVFTLYFGLKPIVVLHGYEAVKEALIDLGEEFSGRGIFPLAERANRGFGIVFSNGKKWKEIRRFSLMTLRNFGMGKRSIEDRVQEEARCLVEELRKTKASPCDPTFILGCAPCNVICSIIIFHKRFDYKDQQFLNLMEKLNENIKILSSPWIQICNNFSPIIDYFPGTHNKLLKNVAFMKSYILEKVKEHQESMDMNNPQDFIDCFLMKMEKEKHNQPSEFTIESLENTAVDLFGAGTETTSTTLRYALLLLLKHPEVTAKVQEEIERVIGRNRSPCMQDRSHMPYTDAVVHEVQRYIDLLPTSLPHAVTCDIKFRNYLIPKGTTILISLTSVLHDNKEFPNPEMFDPHHFLDEGGNFKKSKYFMPFSAGKRICVGEALAGMELFLFLTSILQNFNLKSLVDPKNLDTTPVVNGFASVPPPFYQLCFIPV"
CYP2E1 = "MSALGVTVALLVWAAFLLLVSMWRQVHSSWNLPPGPFPLPIIGNLFQLELKNIPKSFTRLAQRFGPVFTLYVGSQRMVVMHGYKAVKEALLDYKDEFSGRGDLPAFHAHRDRGIIFNNGPTWKDIRRFSLTTLRNYGMGKQGNESRIQREAHFLLEALRKTQGQPFDPTFLIGCAPCNVIADILFRKHFDYNDEKFLRLMYLFNENFHLLSTPWLQLYNNFPSFLHYLPGSHRKVIKNVAEVKEYVSERVKEHHQSLDPNCPRDLTDCLLVEMEKEKHSAERLYTMDGITVTVADLFFAGTETTSTTLRYGLLILMKYPEIEEKLHEEIDRVIGPSRIPAIKDRQEMPYMDAVVHEIQRFITLVPSNLPHEATRDTIFRGYLIPKGTVVVPTLDSVLYDNQEFPDPEKFKPEHFLNENGKFKYSDYFKPFSTGKRVCAGEGLARMELFLLLCAILQHFNLKPLVDPKDIDLSPIHIGFGCIPPRYKLCVIPRS"
CYP1A1 = "MLFPISMSATEFLLASVIFCLVFWVIRASRPQVPKGLKNPPGPWGWPLIGHMLTLGKNPHLALSRMSQQYGDVLQIRIGSTPVVVLSGLDTIRQALVRQGDDFKGRPDLYTFTLISNGQSMSFSPDSGPVWAARRRLAQNGLKSFSIASDPASSTSCYLEEHVSKEAEVLISTLQELMAGPGHFNPYRYVVVSVTNVICAICFGRRYDHNHQELLSLVNLNNNFGEVVGSGNPADFIPILRYLPNPSLNAFKDLNEKFYSFMQKMVKEHYKTFEKGHIRDITDSLIEHCQEKQLDENANVQLSDEKIINIVLDLFGAGFDTVTTAISWSLMYLVMNPRVQRKIQEELDTVIGRSRRPRLSDRSHLPYMEAFILETFRHSSFVPFTIPHSTTRDTSLKGFYIPKGRCVFVNQWQINHDQKLWVNPSEFLPERFLTPDGAIDKVLSEKVIIFGMGKRKCIGETIARWEVFLFLAILLQRVEFSVPLGVKVDMTPIYGLTMKHACCEHFQMQLRS"

# Non-CYP enzyme sequences (UniProt)
LYSOZYME = "MRSLLILVLCFLPLAALGKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
TRYPSIN = "MKTFIFLALLGAAVAFPVDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"
CARBONIC_ANHYDRASE = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
ADH = "MSTAGKVIKCKAAVLWEEKKPFSIEEVEVAPPKAHEVRIKMVATGICRSDDHVVSGTLVTPLPVIAGHEAAGIVESIGEGVTTVRPGDKVIPLFTPQCGKCRVCKHPEGNFCLKNDLSMPPRGTMQDGTSRFTCRGKPIHHFLGTSTFSQYTVVDEISVAKIDAASPLEKVCLIGCGFSTGYGSAVKVAKVTQGSTCAVFGLGGVGLSVIMGCKAAGAARIIGVDINKDKFAKAKEVGATECVNPQDYKKPIQEVLTEMSNGGVDFSFEVIGRLDTMVTALSCCQEAYGVSVIVGVPPDSQNLSMNPMLLLSGRTWKGAIFGGFKSKDSVPKLVADFMAKKFALDPLITHVLPFEKINEGFDLLRSGESIRTILTF"
ACHE = "MRPPQCLLHTPSLASPLLLLLLWLLGGGVGAEGREDAELLVTVRGGRLRGIRLKTPGGPVSAFLGIPFAEPPMGPRRFLPPEPKQPWSGVVDATTFQSVCYQYVDTLYPGFEGTEMWNPNRELSEDCLYLNVWTPYPRPTSPTPVLVWIYGGGFYSGASSLDVYDGRFLVQAERTVLVSMNYRVGAFGFLALPGSREAPGNVGLLDQRLALQWVQENVAAFGGDPTSVTLFGESAGAASVGMHLLSPPSRGLFHRAVLQSGAPNGPWATVGMGEARRRATQLAHLVGCPPGGTGGNDTELVACLRTRPAQVLVNHEWHVLPQESVFRFSFVPVVDGDFLSDTPEALINAGDFHGLQVLVGVVKDEGSYFLVYGAPGFSKDNESLISRAEFLAGVRVGVPQVSDLAAEAVVLHYTDWLHPEDPARLREALSDVVGDHNVVCPVAQLAGRLAAQGARVYAYVFEHRASTLSWPLWMGVPHGYEIEFIFGIPLDPSRNYTAEEKIFAQRLMRYWANFARTGDPNEPRDPKAPQWPPYTAGAQQYVSLDLRPLEVRRGLRAQACAFWNRFLPKLLSATDTLDEAERQWKAEFHRWSSYMVHWKNQFDHYSKQDRCSDL"
CATALASE = "MSTSDDIHNTTASGKCPFHQGGHDQSAGAGTTTRDWWPNQLRVDLLNQHSNRSNPLGEDFDYRKEFSKLDYYGLKKDLKALLTESQPWWPADWGSYAGLFIRMAWHHAGTYRSIDGRGGAGRGQQRFAPLNSWPDNVSLDKARRLLWPIKQKYGQKISWADLFILAGNVALENSGSRTFGFGAGREDVWEPDLDVNWGDEKAWLTHRHPEALAKAPLGATEMGLIYVNPEGPDHSGEPLSAAAAIRATFGNMGMNDEETVALIAGGHTLGKTHGAGPTSNVGPDPEAAPIEEQGLGWASTYGSGVGADAITSGLEVVWTQTPTQWSNYFFENLFKYEWVQTRSPAGAIQFEAVDAPEIIPSDPFDPSKKRKPTMLVTDLTLRFDPEFEKISRRFLNDPQAFNEAFARAWFKLTHRDMGPKSRYIGPEVPKEDLIWQDPLPQPIYNPTEQDIDLKFAIADSGLLSVSELVSVAWARASTFRGGDKRGGANGARLAMLPQRDWDVNAAAVRALPALEKIQKESGKASLADIIVLAGVVGVEKAASAAGLSIHVPFAPGRVDARQDQTDIEMFELLEPIADGFRNYRARLDVSTTESLLIDKAQQQLTLTAPEMTALVGGMRVLGANFDGSKNGVFTDRVGVLSNDFFVNLLDMRYEWKATDESKELFEGRDRETGEVKFTASRADLVFGSNSVLRAVAEVYASSDAHEKFVKDFVAAWVKVMNLDRFDLL"
MAO_A = "MENQEKASIAGHMFDVVVIGGGISGLSAAKLLTEYGVSVLVLEARDRVGGRTYTIRNEHVDYVDVGGAYVGPTQNRILRLSKELGIETYKVNVSERLVQYVKGKTYPFRGAFPPVWNPIAYLDYNNLWRTIDNMGKEIPTDAPWEAQHADKWDKMTMKELIDKICWTKTARRFAYLFVNINVTSEPHEVSALWFLWYVKQCGGTTRIFSVTNGGQERKFVGGSGQVSERIMDLLGDQVKLNHPVTHVDQSSDNIIIETLNHEHYECKYVINAIPPTLTAKIHFRPELPAERNQLIQRLPMGAVIKCMMYYKEAFWKKKDYCGCMIIEDEDAPISITLDDTKPDGSLPAIMGFILARKADRLAKLHKEIRKKKICELYAKVLGSQEALHPVHYEEKNWCEEQYSGGCYTAYFPPGIMTQYGRVIRQPVGRIFFAGTETATKWSGYMEGAVEAGERAAREVLNGLGKVTEKDIWVQEPESKDVPAVEITHTFWERNLPSVSGLLKIIGFSTSVTALGFVLYKYKLLPRS"
LIPASE_B = "MKLLSLTGVAGVLATCVAATPLVKRLPSGSDPAFSQPKSVLDAGLTCQGASPSSVSKPILLLVPGTGTTGPQSFDSNWIPLSTQLGYTPCWISPPPFMLNDTQVNTEYMVNAITALYAGSGNNKLPVLTWSQGGLVAQWGLTFFPSIRSKVDRLMAFAPDYKGTVLAGPLDALAVSAPSVWQQTTGSALTTALRNAGGLTQIVPTTNLYSATDEIVQPQVSNSPLDSSYLFNGKNVQAQAVCGPLFVIDHAGSLTSQFSYVVGRSALRSTTGQARSADYGITDCNPLPANDLTPEQKVAAAAALLAP"


# ── 20 Test Pairs ───────────────────────────────────────────────────────────

TEST_CASES = [
    # === POSITIVE: CYP probe substrates ===
    {
        "id": 1, "label": 1,
        "enzyme": "CYP3A4", "substrate": "Midazolam",
        "catalytic_type": "CYP oxidation (1'-hydroxylation)",
        "sequence": CYP3A4,
        "smiles": "CC1=NC=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4F",
    },
    {
        "id": 2, "label": 1,
        "enzyme": "CYP2D6", "substrate": "Dextromethorphan",
        "catalytic_type": "CYP oxidation (O-demethylation)",
        "sequence": CYP2D6,
        "smiles": "CN1CCC23CCCCC2C1CC4=C3C=C(C=C4)OC",
    },
    {
        "id": 3, "label": 1,
        "enzyme": "CYP1A2", "substrate": "Caffeine",
        "catalytic_type": "CYP oxidation (N3-demethylation)",
        "sequence": CYP1A2,
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    },
    {
        "id": 4, "label": 1,
        "enzyme": "CYP2C9", "substrate": "S-Warfarin",
        "catalytic_type": "CYP oxidation (7-hydroxylation)",
        "sequence": CYP2C9,
        "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
    },
    {
        "id": 5, "label": 1,
        "enzyme": "CYP2E1", "substrate": "Chlorzoxazone",
        "catalytic_type": "CYP oxidation (6-hydroxylation)",
        "sequence": CYP2E1,
        "smiles": "C1=CC2=C(C=C1Cl)NC(=O)O2",
    },
    {
        "id": 6, "label": 1,
        "enzyme": "CYP1A1", "substrate": "Benzo[a]pyrene",
        "catalytic_type": "CYP oxidation (7,8-epoxidation)",
        "sequence": CYP1A1,
        "smiles": "C1=CC2=C3C(=C1)C=CC4=CC=CC5=C4C3=C(C=C5)C=C2",
    },
    # === POSITIVE: Non-CYP classic reactions ===
    {
        "id": 7, "label": 1,
        "enzyme": "Lysozyme", "substrate": "NAG4 (chitotetraose)",
        "catalytic_type": "Glycoside hydrolysis (beta-1,4)",
        "sequence": LYSOZYME,
        "smiles": "CC(=O)NC1C(C(C(OC1O)CO)OC2C(C(C(OC2O)CO)OC3C(C(C(OC3O)CO)OC4C(C(C(OC4O)CO)O)NC(=O)C)O)NC(=O)C)O",
    },
    {
        "id": 8, "label": 1,
        "enzyme": "Trypsin", "substrate": "BAEE",
        "catalytic_type": "Serine protease (Arg ester hydrolysis)",
        "sequence": TRYPSIN,
        "smiles": "CCOC(=O)C(CCCN=C(N)N)NC(=O)C1=CC=CC=C1",
    },
    {
        "id": 9, "label": 1,
        "enzyme": "Carbonic Anhydrase II", "substrate": "CO2",
        "catalytic_type": "Zinc metalloenzyme (CO2 hydration)",
        "sequence": CARBONIC_ANHYDRASE,
        "smiles": "O=C=O",
    },
    {
        "id": 10, "label": 1,
        "enzyme": "Alcohol Dehydrogenase", "substrate": "Ethanol",
        "catalytic_type": "Zinc-dependent oxidation (NAD+)",
        "sequence": ADH,
        "smiles": "CCO",
    },
    {
        "id": 11, "label": 1,
        "enzyme": "Acetylcholinesterase", "substrate": "Acetylcholine",
        "catalytic_type": "Serine esterase (ACh hydrolysis)",
        "sequence": ACHE,
        "smiles": "CC(=O)OCC[N+](C)(C)C",
    },
    {
        "id": 12, "label": 1,
        "enzyme": "Catalase", "substrate": "H2O2",
        "catalytic_type": "Heme peroxidase (disproportionation)",
        "sequence": CATALASE,
        "smiles": "OO",
    },
    {
        "id": 13, "label": 1,
        "enzyme": "MAO-A", "substrate": "Serotonin",
        "catalytic_type": "FAD-dependent amine oxidation",
        "sequence": MAO_A,
        "smiles": "C1=CC2=C(C=C1O)C(=CN2)CCN",
    },
    {
        "id": 14, "label": 1,
        "enzyme": "Lipase B (CALB)", "substrate": "p-Nitrophenyl butyrate",
        "catalytic_type": "Serine hydrolase (ester hydrolysis)",
        "sequence": LIPASE_B,
        "smiles": "CCCC(=O)OC1=CC=C(C=C1)[N+](=O)[O-]",
    },
    # === NEGATIVE: Mismatched enzyme-substrate pairs ===
    {
        "id": 15, "label": 0,
        "enzyme": "CYP3A4", "substrate": "Glucose",
        "catalytic_type": "NEGATIVE: hydrophilic sugar, not CYP substrate",
        "sequence": CYP3A4,
        "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O",
    },
    {
        "id": 16, "label": 0,
        "enzyme": "Lysozyme", "substrate": "Testosterone",
        "catalytic_type": "NEGATIVE: steroid, no glycosidic bonds",
        "sequence": LYSOZYME,
        "smiles": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    },
    {
        "id": 17, "label": 0,
        "enzyme": "Trypsin", "substrate": "Cholesterol",
        "catalytic_type": "NEGATIVE: lipid, no peptide bonds",
        "sequence": TRYPSIN,
        "smiles": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    },
    {
        "id": 18, "label": 0,
        "enzyme": "CYP2E1", "substrate": "ATP",
        "catalytic_type": "NEGATIVE: nucleotide, too polar for CYP",
        "sequence": CYP2E1,
        "smiles": "C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N",
    },
    {
        "id": 19, "label": 0,
        "enzyme": "MAO-A", "substrate": "Sucrose",
        "catalytic_type": "NEGATIVE: disaccharide, no amine group",
        "sequence": MAO_A,
        "smiles": "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O",
    },
    {
        "id": 20, "label": 0,
        "enzyme": "Alcohol Dehydrogenase", "substrate": "Cellobiose",
        "catalytic_type": "NEGATIVE: disaccharide, too bulky for ADH",
        "sequence": ADH,
        "smiles": "C(C1C(C(C(C(O1)OC2C(OC(C(C2O)O)O)CO)O)O)O)O",
    },
]


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    # Add parent to path for imports
    zz_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if zz_root not in sys.path:
        sys.path.insert(0, os.path.dirname(zz_root))

    from zhuangzhou.enzyme_specificity import predict_specificity

    results = []
    total_start = time.time()

    print("=" * 80)
    print("Enzyme Specificity Model Benchmark (20 pairs)")
    print("=" * 80)

    for tc in TEST_CASES:
        print(f"\n[{tc['id']:2d}/20] {tc['enzyme']} + {tc['substrate']}")
        print(f"       Type: {tc['catalytic_type']}")
        print(f"       Seq length: {len(tc['sequence'])}, SMILES: {tc['smiles'][:50]}...")

        t0 = time.time()
        result = predict_specificity(
            protein_sequence=tc['sequence'],
            substrate_smiles=tc['smiles'],
            model_idx=0,
            timeout=600,
        )
        elapsed = time.time() - t0

        if 'error' in result:
            print(f"       ERROR: {result['error']}")
            print(f"       Hint: {result.get('hint', 'N/A')[:200]}")
            results.append({**tc, 'probability': None, 'logit': None,
                            'error': result['error'], 'time': elapsed})
        else:
            prob = result['probability']
            logit = result['logit']
            label = tc['label']
            pred = 1 if prob >= 0.5 else 0
            correct = "OK" if pred == label else "WRONG"
            print(f"       Prob: {prob:.6f}  Logit: {logit:.4f}  "
                  f"Pred: {pred}  Label: {label}  [{correct}]  ({elapsed:.1f}s)")
            results.append({**tc, 'probability': prob, 'logit': logit,
                            'predicted': pred, 'correct': pred == label, 'time': elapsed})

    total_time = time.time() - total_start

    # ── Summary Report ──
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    valid = [r for r in results if r.get('probability') is not None]
    errors = [r for r in results if r.get('probability') is None]

    if valid:
        correct_count = sum(1 for r in valid if r.get('correct', False))
        accuracy = correct_count / len(valid)

        # Separate positive and negative
        pos = [r for r in valid if r['label'] == 1]
        neg = [r for r in valid if r['label'] == 0]
        tp = sum(1 for r in pos if r.get('predicted') == 1)
        fn = sum(1 for r in pos if r.get('predicted') == 0)
        tn = sum(1 for r in neg if r.get('predicted') == 0)
        fp = sum(1 for r in neg if r.get('predicted') == 1)

        print(f"\nTotal valid predictions: {len(valid)}/{len(results)}")
        print(f"Overall Accuracy: {correct_count}/{len(valid)} = {accuracy:.1%}")
        print(f"  True Positives:  {tp}/{len(pos)}  (sensitivity/recall)")
        print(f"  False Negatives: {fn}/{len(pos)}")
        print(f"  True Negatives:  {tn}/{len(neg)}  (specificity)")
        print(f"  False Positives: {fp}/{len(neg)}")
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"  Precision: {precision:.1%}")
        if tp + fn > 0:
            recall = tp / (tp + fn)
            print(f"  Recall:    {recall:.1%}")

        print(f"\nTotal time: {total_time:.1f}s (avg {total_time/len(results):.1f}s/pair)")

        # Detailed table
        print("\n" + "-" * 100)
        print(f"{'ID':>3} {'Enzyme':<25} {'Substrate':<22} {'Label':>5} {'Prob':>8} {'Logit':>8} {'Pred':>5} {'OK?':>4}")
        print("-" * 100)
        for r in valid:
            ok = "Y" if r.get('correct') else "N"
            print(f"{r['id']:>3} {r['enzyme']:<25} {r['substrate']:<22} "
                  f"{r['label']:>5} {r['probability']:>8.6f} {r['logit']:>8.4f} "
                  f"{r.get('predicted', '?'):>5} {ok:>4}")
        print("-" * 100)

        # Positive pair probability distribution
        if pos:
            probs_pos = [r['probability'] for r in pos]
            print(f"\nPositive pairs probability: min={min(probs_pos):.6f}, "
                  f"max={max(probs_pos):.6f}, mean={sum(probs_pos)/len(probs_pos):.6f}")
        if neg:
            probs_neg = [r['probability'] for r in neg]
            print(f"Negative pairs probability: min={min(probs_neg):.6f}, "
                  f"max={max(probs_neg):.6f}, mean={sum(probs_neg)/len(probs_neg):.6f}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for r in errors:
            print(f"  [{r['id']}] {r['enzyme']} + {r['substrate']}: {r.get('error', 'unknown')}")

    # Save JSON
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        # Remove sequence from output (too long)
        clean = [{k: v for k, v in r.items() if k != 'sequence'} for r in results]
        json.dump({'results': clean, 'accuracy': accuracy if valid else None,
                   'total_time': total_time}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {report_path}")


if __name__ == '__main__':
    main()
