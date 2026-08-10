import csv
import os


def gerar_glossario_astronomico():
    print("Gerando Glossário de Termos Científicos e Definições...")

    termos = [
    {
        "termo": "Exoplaneta",
        "def_simples": "Um exoplaneta, ou planeta extrassolar, é qualquer mundo que orbita uma estrela diferente do nosso Sol, encontrando-se fora do nosso Sistema Solar. Desde a primeira descoberta confirmada na década de 1990, milhares foram identificados, variando desde gigantes gasosos maiores que Júpiter até pequenos mundos rochosos do tamanho da Terra, alguns com potencial para abrigar vida.",
        "def_tecnica": "Corpo de massa sub-estelar (< 13 massas de Júpiter) que orbita uma estrela, remanescente estelar ou anã marrom, sem pressão e temperatura suficientes para sustentar fusão nuclear de deutério em seu núcleo.",
        "unidade": "Massas Terrestres (M⊕) / Massas Jovianas (MJ)",
        "fonte": "IAU (International Astronomical Union) / NASA Exoplanet Archive"
    },
    {
        "termo": "Parsec (pc)",
        "def_simples": "O parsec é uma das unidades de medida de distância mais usadas pelos astrônomos profissionais para mapear o espaço profundo. Ele equivale a cerca de 3,26 anos-luz (ou 30 trilhões de quilômetros). A palavra é uma abreviação de 'paralaxe de um segundo', referindo-se a um método trigonométrico de medir distâncias estelares.",
        "def_tecnica": "Distância na qual o raio da órbita da Terra (1 Unidade Astronômica) subtende um ângulo de paralaxe de exatamente um segundo de arco. 1 pc ≈ 3.085677581 × 10^16 metros.",
        "unidade": "pc / kpc / Mpc",
        "fonte": "IAU (International Astronomical Union)"
    },
    {
        "termo": "Método de Trânsito",
        "def_simples": "É a técnica mais bem-sucedida da atualidade para encontrar exoplanetas, usada por telescópios como Kepler e TESS. Ela consiste em observar fixamente uma estrela e medir minúsculas quedas regulares no seu brilho, que ocorrem quando um planeta passa (transita) exatamente na frente dela, bloqueando uma fração de sua luz.",
        "def_tecnica": "Detecção indireta via fotometria de alta precisão que quantifica o decréscimo periódico no fluxo luminoso estelar, permitindo calcular o raio orbital, o raio do planeta e a inclinação orbital.",
        "unidade": "Variação de Magnitude / Curva de Luz (%)",
        "fonte": "NASA Exoplanet Science Institute"
    },
    {
        "termo": "Espectroscopia de Transmissão",
        "def_simples": "É a técnica que permite descobrir do que é feito o ar de mundos alienígenas. Quando um exoplaneta passa em frente à sua estrela, a luz estelar atravessa a atmosfera do planeta. Os gases presentes ali (como água ou metano) absorvem cores específicas da luz, deixando uma 'impressão digital' no espectro que os telescópios capturam e analisam.",
        "def_tecnica": "Análise da variação do fluxo estelar dependente do comprimento de onda durante um trânsito. As assinaturas de absorção refletem a composição química, nuvens e opacidade atmosférica do exoplaneta.",
        "unidade": "Profundidade de trânsito espectral (ppm)",
        "fonte": "Astrofísica Exoplanetária / James Webb Space Telescope (NASA/ESA)"
    },
    {
        "termo": "Velocidade Radial",
        "def_simples": "Também conhecida como 'método do bamboleio', esta técnica detecta exoplanetas observando a influência gravitacional que eles exercem sobre suas estrelas. O planeta puxa a estrela levemente para frente e para trás, causando uma mudança na cor da luz emitida pela estrela (efeito Doppler), revelando não só a presença do planeta, mas a sua massa.",
        "def_tecnica": "Medição astrométrica do desvio Doppler (redshift/blueshift) nas linhas espectrais de uma estrela, provocado pela oscilação no centro de massa (baricentro) do sistema estelar-planetário.",
        "unidade": "m/s (metros por segundo)",
        "fonte": "Observatório Europeu do Sul (ESO) / HARPS"
    },
    {
        "termo": "Zona Habitável",
        "def_simples": "Também chamada de 'Zona Cachinhos Dourados', é a região ao redor de uma estrela onde não é nem quente nem frio demais. Se um planeta rochoso estiver localizado nessa faixa de distância, ele poderá manter água em estado líquido em sua superfície, o que é considerado o pré-requisito fundamental para a vida como a conhecemos.",
        "def_tecnica": "Intervalo de distâncias orbitais de uma estrela onde o fluxo de radiação mantém a temperatura de equilíbrio planetário entre 273 K e 373 K, assumindo pressão atmosférica adequada.",
        "unidade": "Unidades Astronômicas (AU)",
        "fonte": "Planetary Habitability Laboratory (PHL)"
    },
    {
        "termo": "Teoria da Relatividade Geral",
        "def_simples": "Formulada por Albert Einstein em 1915, esta teoria revolucionou a ciência ao propor que a gravidade não é apenas uma força de atração invisível, mas sim a curvatura do próprio tecido do espaço e do tempo (espaço-tempo). Corpos massivos, como estrelas, criam 'buracos' nesse tecido, ditando o caminho que os planetas e a própria luz devem seguir.",
        "def_tecnica": "Teoria geométrica da gravitação onde a curvatura do espaço-tempo está diretamente relacionada à energia e ao momento da matéria e radiação, descrita rigorosamente pelas Equações de Campo de Einstein.",
        "unidade": "N/A (Geometria do Espaço-Tempo)",
        "fonte": "Artigos Originais de Albert Einstein (1915) / Fundamentos de Astrofísica"
    },
    {
        "termo": "Buraco Negro",
        "def_simples": "Um buraco negro é uma região do cosmos onde a gravidade é tão violenta que nada, absolutamente nada, incluindo a luz, consegue escapar de sua atração. Eles costumam nascer do colapso de estrelas gigantes que morreram e explodiram. O limite invisível em torno dele é chamado de 'horizonte de eventos', o ponto sem volta.",
        "def_tecnica": "Solução das equações de Einstein caracterizada por uma singularidade gravitacional central (ou anel, em buracos negros em rotação) cercada por um horizonte de eventos onde a velocidade de escape excede c (velocidade da luz).",
        "unidade": "Massas Solares (M☉)",
        "fonte": "Observatório de Raios-X Chandra / Event Horizon Telescope (EHT)"
    },
    {
        "termo": "Ondas Gravitacionais",
        "def_simples": "As ondas gravitacionais são ondulações invisíveis que viajam à velocidade da luz, comprimindo e esticando o espaço-tempo por onde passam. Elas são geradas pelos eventos mais violentos do universo, como a colisão de dois buracos negros ou estrelas de nêutrons. Detectá-las permite 'ouvir' o cosmos de uma forma totalmente nova.",
        "def_tecnica": "Perturbações na métrica do espaço-tempo que se propagam como ondas transversais, geradas por massas submetidas à aceleração não-esférica e possuindo momento de quadrupolo não nulo.",
        "unidade": "Amplitude de deformação (Strain, sem dimensão)",
        "fonte": "Colaboração LIGO-Virgo-KAGRA"
    },
    {
        "termo": "Matéria Escura",
        "def_simples": "A matéria escura é uma substância fantasma que compõe cerca de 27% do universo. Ela não emite, reflete ou bloqueia a luz, tornando-a totalmente invisível. Contudo, os cientistas sabem que ela existe porque sua imensa gravidade age como uma cola invisível, impedindo que as galáxias girem rápido demais e se despedacem pelo espaço.",
        "def_tecnica": "Forma teórica de matéria não-bariônica que interage muito fracamente com a força eletromagnética. Sua presença é inferida pelas curvas de rotação galáctica, lentes gravitacionais e anisotropias da radiação de fundo.",
        "unidade": "Massa-energia cosmológica (Ωc)",
        "fonte": "Modelo Cosmológico Lambda-CDM / ESA Planck"
    },
    {
        "termo": "Energia Escura",
        "def_simples": "Se a gravidade e a matéria escura tentam puxar e juntar o universo, a Energia Escura é uma força misteriosa que age como uma 'antigravidade', empurrando o espaço para fora e fazendo o universo se expandir cada vez mais rápido. Ela representa quase 68% de tudo o que existe, sendo um dos maiores enigmas não resolvidos da ciência moderna.",
        "def_tecnica": "Densidade de energia intrínseca do vácuo quântico, modelada astrofisicamente como a Constante Cosmológica (Λ), responsável por gerar uma pressão negativa que causa a expansão acelerada do tecido cósmico.",
        "unidade": "Massa-energia cosmológica (ΩΛ)",
        "fonte": "Pesquisas de Supernovas Tipo Ia / NASA WMAP"
    },
    {
        "termo": "Modelo Padrão (Partículas Fundamentais)",
        "def_simples": "É a teoria ou 'tabela periódica definitiva' que descreve os blocos de construção essenciais do universo. O Modelo Padrão divide as partículas em duas famílias principais: os férmions (que formam toda a matéria sólida que conhecemos, como elétrons e quarks) e os bósons (partículas mensageiras que carregam forças, como os fótons que carregam a luz).",
        "def_tecnica": "Teoria quântica de campos baseada nos grupos de simetria de calibre SU(3)×SU(2)×U(1), que descreve matematicamente as interações forte, fraca e eletromagnética, excluindo apenas a gravidade.",
        "unidade": "Elétron-volt (eV / MeV / GeV)",
        "fonte": "CERN / Particle Data Group (PDG)"
    },
    {
        "termo": "Campo e Bóson de Higgs",
        "def_simples": "O Campo de Higgs é um campo de energia invisível que permeia o universo inteiro, responsável por dar massa às partículas. Pense nele como uma piscina de melaço: as partículas que sofrem resistência ao atravessá-lo ganham massa, enquanto as que passam direto (como fótons) não têm massa. A prova física da existência desse campo foi descoberta em 2012 na forma do 'Bóson de Higgs'.",
        "def_tecnica": "Campo escalar quântico que sofre quebra espontânea de simetria (mecanismo BEH). As interações das partículas de calibre e férmions com o valor esperado do vácuo desse campo geram suas massas de repouso.",
        "unidade": "GeV/c² (Massa invariante)",
        "fonte": "Organização Europeia para a Pesquisa Nuclear (CERN)"
    },
    {
        "termo": "Neutrino",
        "def_simples": "Neutrinos são minúsculas partículas subatômicas chamadas de 'partículas fantasmas'. Eles não possuem carga elétrica e quase não têm massa, permitindo que bilhões deles atravessem seu corpo, a Terra inteira e até as estrelas a cada segundo sem interagir com nada. Eles nascem no coração do nosso Sol, nas reações nucleares, e em supernovas catastróficas.",
        "def_tecnica": "Férmions elementares (sabores: elétron, múon, tau) sujeitos apenas à força nuclear fraca e à gravidade. Ocorrem oscilações de sabor em trânsito, comprovando que possuem massa não-nula, violando o Modelo Padrão original.",
        "unidade": "Elétron-volt (eV)",
        "fonte": "IceCube Neutrino Observatory / Super-Kamiokande"
    },
    {
        "termo": "Antimatéria",
        "def_simples": "A antimatéria é como o reflexo no espelho da matéria comum, sendo composta por partículas que têm a mesma massa, mas carga elétrica oposta. Por exemplo, o anti-elétron é positivo em vez de negativo. A característica mais incrível (e perigosa) da antimatéria é que, se ela tocar na matéria comum, ambas se aniquilam instantaneamente em uma violenta explosão de energia pura.",
        "def_tecnica": "Material composto por antipartículas correspondentes (soluções de energia negativa da equação de Dirac), exibindo números quânticos e cargas (elétricas, de cor, sabor) opostos à matéria bariônica ou leptônica convencional.",
        "unidade": "N/A",
        "fonte": "Fábrica de Antimatéria do CERN / NASA Fermi Gamma-ray Space Telescope"
    }
]

    try:
        caminho_script = os.path.abspath(__file__)
    except NameError:
        caminho_script = os.path.abspath(".")

    diretorio_destino = os.path.join(
        os.path.dirname(caminho_script), "..", "..", "data", "dataset"
    )
    os.makedirs(diretorio_destino, exist_ok=True)
    arquivo_csv = os.path.join(
        diretorio_destino, "glossario_astronomico_conceitos.csv"
    )

    with open(arquivo_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "termo_cientifico",
            "definicao_simples",
            "definicao_tecnica",
            "unidade_medida_relacionada",
            "fonte_dados",
            "status_validacao",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in termos:
            writer.writerow({
                "termo_cientifico": t["termo"],
                "definicao_simples": t["def_simples"],
                "definicao_tecnica": t["def_tecnica"],
                "unidade_medida_relacionada": t["unidade"],
                "fonte_dados": t.get("fonte", "Glossário Científico NASA/IAU"),
                "status_validacao": "Validado",
            })
    print(
        f"   -> Sucesso! Arquivo '{arquivo_csv}' gerado com {len(termos)} registros.\n"
    )


if __name__ == "__main__":
    gerar_glossario_astronomico()