import argparse
import json
from translation_memory import Memory, normalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['stats','list','search','candidates','conflicts','set','lock','unlock','delete','export'])
    parser.add_argument('--work',required=True)
    parser.add_argument('values',nargs='*')
    args = parser.parse_args()
    memory = Memory(args.work)
    try:
        if args.command == 'set':
            if len(args.values)!=2: parser.error('set requires Chinese and English')
            memory.set(*args.values)
        elif args.command in ('lock','unlock','delete'):
            if len(args.values)!=1: parser.error('command requires Chinese term')
            with memory.db:
                if args.command=='delete': memory.db.execute('DELETE FROM terms WHERE source=?',(normalize(args.values[0]),))
                else: memory.db.execute('UPDATE terms SET locked=? WHERE source=?',(int(args.command=='lock'),normalize(args.values[0])))
        elif args.command=='export': print(memory.export())
        elif args.command=='stats':
            print({t:memory.db.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in ('terms','term_candidates','term_conflicts','translation_memory')})
        else:
            table={'candidates':'term_candidates','conflicts':'term_conflicts'}.get(args.command,'terms')
            query='SELECT * FROM '+table
            rows=memory.db.execute(query+' WHERE source LIKE ?' if args.command=='search' else query,
                ('%'+args.values[0]+'%',) if args.command=='search' else ()).fetchall()
            print(json.dumps(rows,ensure_ascii=False,indent=2))
    finally: memory.close()
