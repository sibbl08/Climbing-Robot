from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np

@dataclass
class Hold:
    name: str          
    pos: np.ndarray    

    def __repr__(self) -> str:
        return f"Hold(name={self.name}, pos={self.pos})"

    @property
    def is_start(self) -> bool:
        return "start" in self.name

    @property
    def side(self) -> str:
        """
        Devine le côté (left/right) à partir du nom ou de la coordonnée x.
        """
        n = self.name.lower()
        if "_l" in n or "left" in n:
            return "left"
        if "_r" in n or "right" in n:
            return "right"
        
        if self.pos[0] < 0:
            return "left"
        if self.pos[0] > 0:
            return "right"
        return "center"


ROUTE_XML_PATH = (
    Path(__file__).resolve().parents[1] / "Mujoco" / "route_ladder_V1.xml"
)


def load_holds_from_xml(xml_path: Path | str = ROUTE_XML_PATH) -> List[Hold]:
    """
    Lit le fichier MuJoCo XML de la route d'escalade et retourne une liste de Hold.
    """
    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"Route XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    holds: List[Hold] = []

    
    for body in root.iter("body"):
        name = body.get("name")
        pos_str = body.get("pos")

        if name is None or pos_str is None:
            continue 

       
        pos_vals = [float(v) for v in pos_str.split()]
        pos = np.array(pos_vals, dtype=float)

        holds.append(Hold(name=name, pos=pos))

    return holds

def split_start_and_route_holds(holds: List[Hold]) -> Tuple[List[Hold], List[Hold]]:
    """
    Sépare les prises de départ (start) des prises normales de la route.
    """
    start_holds = [h for h in holds if h.is_start]
    route_holds = [h for h in holds if not h.is_start]
    return start_holds, route_holds


def sort_holds_by_height(holds: List[Hold]) -> List[Hold]:
    """
    Trie les prises par hauteur (coordonnée z), du bas vers le haut.
    """
    return sorted(holds, key=lambda h: h.pos[2])

def main():
    print(f"Loading holds from: {ROUTE_XML_PATH}")
    holds = load_holds_from_xml()

    print(f"\nTotal holds found: {len(holds)}")

    start_holds, route_holds = split_start_and_route_holds(holds)
    print(f"  Start holds: {len(start_holds)}")
    print(f"  Route holds: {len(route_holds)}")

    print("\nStart holds:")
    for h in start_holds:
        print(f"  {h.name:15s}  pos={h.pos}")

    print("\nFirst 10 route holds sorted by height:")
    for h in sort_holds_by_height(route_holds)[:10]:
        print(f"  {h.name:15s}  z={h.pos[2]:.3f}  x={h.pos[0]:.3f}")


if __name__ == "__main__":
    main()
